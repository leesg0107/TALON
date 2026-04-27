"""End-to-end mission eval — parallel headless version.

Mirrors eval_mission.py logic exactly, but runs NUM_ENVS missions simultaneously.
Phases: approach(0) → dock(1) → climb(2) → delivery(3) → arrived(4) → done(5)
"""
import sys, os, math
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # ensure relative paths (logs/, etc.) resolve correctly

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import gymnasium as gym

# ============================================================
# Config
# ============================================================
NUM_ENVS = 128
MAX_STEPS = 18000       # 120s at 150Hz
WP_REACH_DIST = 0.30
APPROACH_TIMEOUT = 9000  # 60s
DOCK_TIMEOUT = 1800      # 12s

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = NUM_ENVS
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 120.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.spawn_spread = 0.0
cfg.truncate_on_dock_success = False
cfg.goal_pos_range_xy = 0.0  # box always at pedestal center
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

# Disable auto-reset (same as eval_mission.py)
def _eval_get_dones():
    return (torch.zeros(NUM_ENVS, dtype=torch.bool, device=device),
            torch.zeros(NUM_ENVS, dtype=torch.bool, device=device))
env._get_dones = _eval_get_dones

# ============================================================
# Load models — action space must be Box(-inf, inf) to avoid clipping
# ============================================================
obs_s1 = gym.spaces.Box(-float('inf'), float('inf'), (22,))
obs_s4 = gym.spaces.Box(-float('inf'), float('inf'), (23,))
act_sp = gym.spaces.Box(-float('inf'), float('inf'), (4,))

class FakeEnv:
    def __init__(self, obs_space, n):
        self.observation_space = obs_space
        self.action_space = act_sp
        self.num_envs = n

agent_s1 = build_waypoint_agent(env=FakeEnv(obs_s1, NUM_ENVS), device=device, mode="flight",
                                checkpoint_path="models/flight_best/final_agent.pt")
agent_s1.set_running_mode("eval")

agent_s4 = build_waypoint_agent(env=FakeEnv(obs_s4, NUM_ENVS), device=device, mode="loaded",
                                checkpoint_path="models/loaded_best/best_agent.pt")
agent_s4.set_running_mode("eval")

# ============================================================
# Obs helpers (batch) — noise matches training: 0.03 vel, 0.01 pos
# ============================================================
_VEL_NOISE = 0.03
_POS_NOISE = 0.01

def compute_obs_batch(env, goal, prev_act, eids, loaded=False, payload=None):
    n = len(eids)
    vel_b = env.robot.data.root_lin_vel_b[eids]
    ang_vel_b = env.robot.data.root_ang_vel_b[eids]
    quat_w = env.robot.data.root_quat_w[eids]
    R = quat_to_rot_matrix(quat_w)
    pos_w = env.robot.data.root_pos_w[eids]
    goal_err_w = goal[eids] - pos_w
    goal_b = torch.bmm(R.transpose(1, 2), goal_err_w.unsqueeze(-1)).squeeze(-1)
    R_flat = R.reshape(n, 9)
    prev_norm = prev_act[eids].clone()
    prev_norm[:, :3] /= 8.0
    prev_norm[:, 3] /= math.pi
    vel_b = vel_b + _VEL_NOISE * torch.randn_like(vel_b)
    goal_b = goal_b + _POS_NOISE * torch.randn_like(goal_b)
    obs = torch.cat([vel_b, ang_vel_b, R_flat, goal_b, prev_norm], dim=-1)
    if loaded:
        pe = payload[eids].unsqueeze(-1) + 0.02 * torch.randn(n, 1, device=device)
        obs = torch.cat([obs, pe], dim=-1)
    return obs

# ============================================================
# WP generation (same as eval_mission.py)
# ============================================================
def generate_waypoints(start, target_xy, target_z=2.0, num_wps=3, min_z=0.8):
    waypoints = []
    prev = list(start)
    for i in range(num_wps):
        dx = target_xy[0] - prev[0]
        dy = target_xy[1] - prev[1]
        dz = target_z - prev[2]
        remaining = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        angle = angle_to_target + (torch.rand(1).item() - 0.5) * 1.5
        dist = 0.8 + torch.rand(1).item() * 1.5
        dist = min(dist, remaining + 0.5)
        wp_x = prev[0] + dist * math.cos(angle)
        wp_y = prev[1] + dist * math.sin(angle)
        t = (i + 1) / (num_wps + 1)
        wp_z = prev[2] + t * dz * 0.5 + (torch.rand(1).item() - 0.5) * 1.0
        wp_z = max(wp_z, min_z)
        waypoints.append(torch.tensor([wp_x, wp_y, wp_z], device=device))
        prev = [wp_x, wp_y, wp_z]
    waypoints.append(torch.tensor([target_xy[0], target_xy[1], target_z], device=device))
    return waypoints

# ============================================================
# Per-env state
# ============================================================
# Phases: settle=0, approach=1, dock=2, climb=3, hover_stabilize=4, delivery=5, arrived=6, done=7
# SETTLE: 1-step wait after teleport for write_root_state_to_sim to commit
PHASE_SETTLE, PHASE_APPROACH, PHASE_DOCK, PHASE_CLIMB, PHASE_HOVER_STAB, PHASE_DELIVERY, PHASE_ARRIVED, PHASE_DONE = 0, 1, 2, 3, 4, 5, 6, 7

phase = torch.full((NUM_ENVS,), PHASE_DONE, dtype=torch.long, device=device)
current_goal = torch.zeros(NUM_ENVS, 3, device=device)
prev_action_4d = torch.zeros(NUM_ENVS, 4, device=device)
wp_idx = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
payload_mass = torch.full((NUM_ENVS,), 0.2, device=device)
approach_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
post_wp_wait = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
dock_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
climb_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
hover_stab_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)

MAX_WPS = 5
approach_wps = torch.zeros(NUM_ENVS, MAX_WPS, 3, device=device)
delivery_wps = torch.zeros(NUM_ENVS, MAX_WPS, 3, device=device)
n_approach_wps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
n_delivery_wps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
delivery_target = torch.zeros(NUM_ENVS, 2, device=device)

results = {"total": 0, "full_success": 0, "dock_success": 0}
fail_reasons = {}

# ============================================================
# Setup mission for given env ids
# ============================================================
def setup_mission(eids):
    for eid in eids.tolist():
        origin = env.scene.env_origins[eid]
        # Box at pedestal center (goal_pos_range_xy=0 ensures this from _sample_objects)
        box_pos = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)

        # Reset box
        obj_state = torch.zeros(1, 13, device=device)
        obj_state[0, :3] = box_pos
        obj_state[0, 3] = 1.0
        env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
        env.object_pos[eid] = box_pos

        # Random start 2-3m from box at z=3
        angle = torch.rand(1).item() * 2 * math.pi
        dist = 2.0 + torch.rand(1).item() * 1.0
        start = [box_pos[0].item() + dist * math.cos(angle),
                 box_pos[1].item() + dist * math.sin(angle),
                 origin[2].item() + 3.0]

        root_state = torch.zeros(1, 13, device=device)
        root_state[0, :3] = torch.tensor(start, device=device)
        root_state[0, 3] = 1.0
        env.robot.write_root_state_to_sim(root_state, torch.tensor([eid], device=device))

        # Reset state
        env.contain_hold_count[eid] = 0
        env.step_count[eid] = 0
        if hasattr(env, '_stuck_count'):
            env._stuck_count[eid] = 0
        if hasattr(env, '_recovery_timer'):
            env._recovery_timer[eid] = 0
        if hasattr(env, '_z_integral'):
            env._z_integral[eid] = 0.0
        if hasattr(env, '_dock_hold_z'):
            env._dock_hold_z[eid] = 0.0
        env.attitude_ctrl.reset(torch.tensor([eid], device=device))

        # Approach WPs
        bxy = [box_pos[0].item(), box_pos[1].item()]
        target_z = box_pos[2].item() + 0.5
        wps = generate_waypoints(start, bxy, target_z=target_z, num_wps=3)
        for j, wp in enumerate(wps):
            approach_wps[eid, j] = wp
        n_approach_wps[eid] = len(wps)

        # Delivery target
        delivery_target[eid, 0] = origin[0].item() + (torch.rand(1).item() - 0.5) * 6.0
        delivery_target[eid, 1] = origin[1].item() + (torch.rand(1).item() - 0.5) * 6.0

        # Init phase state — SETTLE first (1 step for teleport to commit)
        phase[eid] = PHASE_SETTLE
        wp_idx[eid] = 0
        current_goal[eid] = approach_wps[eid, 0]
        prev_action_4d[eid] = 0
        approach_steps[eid] = 0
        post_wp_wait[eid] = 0
        dock_steps[eid] = 0
        climb_steps[eid] = 0
        hover_stab_steps[eid] = 0
        if hasattr(setup_mission, '_dock_counted'):
            setup_mission._dock_counted[eid] = False

    env.reset_buf[eids] = False
    env.reset_terminated[eids] = False
    env.reset_time_outs[eids] = False

def record_fail(eid, reason, detail=""):
    results["total"] += 1
    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
    if "climb" in reason and detail:
        print(f"    CLIMB_FAIL env{eid}: {detail}")
    if detail:
        print(f"    [env{eid}] FAIL: {reason} — {detail}")

# ============================================================
# Init
# ============================================================
obs, _ = env_wrapped.reset()
all_ids = torch.arange(NUM_ENVS, device=device)
setup_mission(all_ids)

# Initial settle: dummy steps to commit teleport + settle physics
env.bypass_analytical = True
for _ in range(3):
    env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))
    env.contain_hold_count[:] = 0

# Re-place all boxes on pedestal center
for eid in range(NUM_ENVS):
    origin = env.scene.env_origins[eid]
    bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = bp
    obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
    env.object_pos[eid] = bp

# Commit box placement + one more settle
env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))
env.contain_hold_count[:] = 0

# Now transition all envs from SETTLE → APPROACH
phase[:] = PHASE_APPROACH

TARGET_MISSIONS = 500
print(f"\n{'='*60}")
print(f"  END-TO-END PARALLEL EVAL ({NUM_ENVS} envs, target={TARGET_MISSIONS} missions)")
print(f"  Stage 1: gripper_wp_flight_v6  |  Stage 4: gripper_wp_loaded_v13")
print(f"{'='*60}\n")

# ============================================================
# Main loop
# ============================================================
for step in range(MAX_STEPS * 10):  # enough steps for TARGET_MISSIONS
    if results["total"] + results["full_success"] >= TARGET_MISSIONS:
        break

    pos_w = env.robot.data.root_pos_w
    obj_pos = env.object_pos
    action_8d = torch.zeros(NUM_ENVS, 8, device=device)

    # ---- SETTLE (phase==0) — wait 1 step for teleport to commit ----
    s_mask = (phase == PHASE_SETTLE)
    if s_mask.any():
        s_ids = s_mask.nonzero(as_tuple=False).view(-1)
        action_8d[s_ids] = 0  # zero action during settle
        # Re-place box to ensure it's on pedestal (write committed this step)
        for eid in s_ids.tolist():
            origin = env.scene.env_origins[eid]
            bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
            obj_st = torch.zeros(1, 13, device=device)
            obj_st[0, :3] = bp
            obj_st[0, 3] = 1.0
            env.grasp_object.write_root_state_to_sim(obj_st, torch.tensor([eid], device=device))
            env.object_pos[eid] = bp
            env.contain_hold_count[eid] = 0
        # Transition to approach on NEXT step (after this step commits the teleport)
        phase[s_ids] = PHASE_APPROACH

    # ---- APPROACH (phase==1) ----
    a_mask = (phase == PHASE_APPROACH)
    if a_mask.any():
        a_ids = a_mask.nonzero(as_tuple=False).view(-1)
        approach_steps[a_ids] += 1

        obs_22d = compute_obs_batch(env, current_goal, prev_action_4d, a_ids)
        with torch.no_grad():
            act = agent_s1.act(obs_22d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[a_ids] = act
        action_8d[a_ids, :3] = act[:, :3]
        action_8d[a_ids, 6] = act[:, 3]

        # WP reach
        dist_wp = torch.norm(pos_w[a_ids] - current_goal[a_ids], dim=-1)
        for i, eid in enumerate(a_ids.tolist()):
            if dist_wp[i].item() < WP_REACH_DIST:
                wi = wp_idx[eid].item() + 1
                if wi < n_approach_wps[eid].item():
                    wp_idx[eid] = wi
                    current_goal[eid] = approach_wps[eid, wi]

        # Transition check (relaxed, same as eval_mission.py)
        xy_box = torch.norm(pos_w[a_ids, :2] - obj_pos[a_ids, :2], dim=-1)
        z_above = pos_w[a_ids, 2] - obj_pos[a_ids, 2]
        vel_w = env.robot.data.root_lin_vel_w[a_ids]
        speed = torch.norm(vel_w, dim=-1)
        R_a = quat_to_rot_matrix(env.robot.data.root_quat_w[a_ids])
        tilt = torch.acos(R_a[:, 2, 2].clamp(-1, 1)) * 57.3

        for i, eid in enumerate(a_ids.tolist()):
            last_wp = wp_idx[eid].item() >= n_approach_wps[eid].item()
            timeout = approach_steps[eid].item() > APPROACH_TIMEOUT

            if last_wp:
                post_wp_wait[eid] += 1

            if last_wp or timeout:
                pw = post_wp_wait[eid].item()
                sp = speed[i].item()
                ti = tilt[i].item()
                xy = xy_box[i].item()
                za = z_above[i].item()

                stable = sp < 2.0 and ti < 30 and xy < 0.8
                if pw > 750:
                    stable = xy < 1.5 and za > 0 and za < 2.0

                if stable:
                    phase[eid] = PHASE_DOCK
                    dock_steps[eid] = 0
                    print(f"    [env{eid}] DOCK ENTRY: v={sp:.2f} tilt={ti:.0f} xy={xy:.2f} z={za:.2f} wait={pw/150:.1f}s")
                elif timeout:
                    if xy < 2.0 and za > 0:
                        phase[eid] = PHASE_DOCK
                        dock_steps[eid] = 0
                        print(f"    [env{eid}] DOCK ENTRY (TIMEOUT): v={sp:.2f} tilt={ti:.0f} xy={xy:.2f} z={za:.2f}")
                    else:
                        phase[eid] = PHASE_DONE
                        record_fail(eid, "approach_timeout_too_far")
                else:
                    current_goal[eid] = approach_wps[eid, n_approach_wps[eid].item() - 1]

    # ---- DOCK (phase==2) ----
    d_mask = (phase == PHASE_DOCK)
    if d_mask.any():
        d_ids = d_mask.nonzero(as_tuple=False).view(-1)
        action_8d[d_ids] = 0  # PD controls
        dock_steps[d_ids] += 1

        for i, eid in enumerate(d_ids.tolist()):
            contain = env.contain_hold_count[eid].item()
            box_z_local = obj_pos[eid, 2].item() - env.scene.env_origins[eid, 2].item()

            # Box fell
            if box_z_local < 0.30 and contain < 150:
                phase[eid] = PHASE_DONE
                record_fail(eid, "box_fell_during_dock")
                continue

            if contain >= 325:
                box_drone_dist = torch.norm(pos_w[eid] - obj_pos[eid]).item()
                if box_drone_dist < 0.30:
                    # Grasped → climb (only count first success per mission)
                    drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[eid]
                    env.attitude_ctrl.mass[eid] = drone_mass + 0.2
                    phase[eid] = PHASE_CLIMB
                    climb_steps[eid] = 0
                else:
                    # Grip missed → reset contain, gripper re-opens, PD retries
                    env.contain_hold_count[eid] = 0
                    if hasattr(env, '_stuck_count'):
                        env._stuck_count[eid] = 0
                continue

            # Dock timeout
            if dock_steps[eid].item() > DOCK_TIMEOUT:
                xy = torch.norm(pos_w[eid, :2] - obj_pos[eid, :2]).item()
                za = pos_w[eid, 2].item() - obj_pos[eid, 2].item()
                if contain >= 325:
                    reason = f"dock_timeout_gripper_closed(c={contain})"
                elif contain > 50:
                    reason = f"dock_timeout_partial(c={contain})"
                elif xy > 0.15:
                    reason = f"dock_timeout_xy(xy={xy:.2f})"
                elif za > 0.30:
                    reason = f"dock_timeout_high(z={za:.2f})"
                else:
                    reason = f"dock_timeout_stuck(xy={xy:.2f},z={za:.2f},c={contain})"
                phase[eid] = PHASE_DONE
                record_fail(eid, reason)

    # ---- CLIMB (phase==3) — PD auto-climbs, wait for altitude ----
    c_mask = (phase == PHASE_CLIMB)
    if c_mask.any():
        c_ids = c_mask.nonzero(as_tuple=False).view(-1)
        action_8d[c_ids] = 0  # PD controls climb
        climb_steps[c_ids] += 1

        for i, eid in enumerate(c_ids.tolist()):
            cur_z = pos_w[eid, 2].item()
            box_z = obj_pos[eid, 2].item()
            box_z_local = box_z - env.scene.env_origins[eid, 2].item()
            bd = torch.norm(pos_w[eid] - obj_pos[eid]).item()
            cs = climb_steps[eid].item()

            # Early detect 1: box stayed on pedestal, drone flew up alone → retry
            if cs > 30 and box_z_local < 0.60 and cur_z - box_z > 0.25:
                env.contain_hold_count[eid] = 0
                if hasattr(env, '_stuck_count'):
                    env._stuck_count[eid] = 0
                # _dock_counted stays True — same mission, don't re-count
                phase[eid] = PHASE_DOCK
                dock_steps[eid] = 0
                results["grasp_retries"] = results.get("grasp_retries", 0) + 1
                continue

            # Early detect 2: drone+box fell together to ground → mission failed
            drone_z_local = cur_z - env.scene.env_origins[eid, 2].item()
            if cs > 60 and drone_z_local < 0.30:
                phase[eid] = PHASE_DONE
                record_fail(eid, "climb_failed", f"ground_crash drone_z={drone_z_local:.2f} box_z={box_z_local:.2f}")
                continue

            if cur_z > 1.0:
                # Climb done → delivery directly
                cur = pos_w[eid].tolist()
                dt_xy = [delivery_target[eid, 0].item(), delivery_target[eid, 1].item()]
                del_wps = generate_waypoints(cur, dt_xy, target_z=2.5, num_wps=3, min_z=1.5)
                for j, wp in enumerate(del_wps):
                    delivery_wps[eid, j] = wp
                n_delivery_wps[eid] = len(del_wps)
                wp_idx[eid] = 0
                current_goal[eid] = delivery_wps[eid, 0]
                prev_action_4d[eid] = 0
                phase[eid] = PHASE_DELIVERY
            elif cs > 1500:
                phase[eid] = PHASE_DONE
                record_fail(eid, "climb_timeout", f"z={cur_z:.2f} bd={bd:.2f} box_z_l={box_z_local:.2f}")
            elif cur_z < 0.05:
                phase[eid] = PHASE_DONE
                record_fail(eid, "climb_crashed", f"z={cur_z:.2f}")
            elif bd > 0.50:
                phase[eid] = PHASE_DONE
                origin_z = env.scene.env_origins[eid, 2].item()
                drone_z_l = cur_z - origin_z
                record_fail(eid, "climb_box_lost", f"bd={bd:.2f} drone_z={drone_z_l:.2f} box_z={box_z_local:.2f}")

    # ---- HOVER STABILIZE (phase==4) — PD holds position, drone settles before RL takes over ----
    h_mask = (phase == PHASE_HOVER_STAB)
    if h_mask.any():
        h_ids = h_mask.nonzero(as_tuple=False).view(-1)
        # Use loaded-flight RL model for hover (goal = current position → near-zero action)
        # This lets the RL model "warm up" with gentle commands
        for eid in h_ids.tolist():
            hover_stab_steps[eid] += 1
            current_goal[eid] = pos_w[eid].clone()  # goal = current pos → hover

        obs_23d = compute_obs_batch(env, current_goal, prev_action_4d, h_ids, loaded=True, payload=payload_mass)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[h_ids] = act
        action_8d[h_ids, :3] = act[:, :3]
        action_8d[h_ids, 6] = act[:, 3]

        # Transition to delivery after 1.5s (225 steps) of stable hover
        for eid in h_ids.tolist():
            hs = hover_stab_steps[eid].item()
            if hs >= 225:
                wp_idx[eid] = 0
                current_goal[eid] = delivery_wps[eid, 0]
                prev_action_4d[eid] = 0
                phase[eid] = PHASE_DELIVERY

    # ---- DELIVERY (phase==5) ----
    v_mask = (phase == PHASE_DELIVERY)
    if v_mask.any():
        v_ids = v_mask.nonzero(as_tuple=False).view(-1)
        obs_23d = compute_obs_batch(env, current_goal, prev_action_4d, v_ids, loaded=True, payload=payload_mass)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[v_ids] = act
        action_8d[v_ids, :3] = act[:, :3]
        action_8d[v_ids, 6] = act[:, 3]

        dist_wp = torch.norm(pos_w[v_ids] - current_goal[v_ids], dim=-1)
        for i, eid in enumerate(v_ids.tolist()):
            # Box dropped check
            box_z_l = obj_pos[eid, 2].item() - env.scene.env_origins[eid, 2].item()
            drone_z_l = pos_w[eid, 2].item() - env.scene.env_origins[eid, 2].item()
            box_dist = torch.norm(pos_w[eid] - obj_pos[eid]).item()
            if box_z_l < 0.30 or box_dist > 0.50:
                if box_z_l < 0.30 and drone_z_l < 0.30:
                    detail = f"drone+box crashed drone_z={drone_z_l:.2f} box_z={box_z_l:.2f}"
                elif box_z_l < 0.30:
                    detail = f"box fell drone_z={drone_z_l:.2f} box_z={box_z_l:.2f} dist={box_dist:.2f}"
                else:
                    detail = f"box drifted dist={box_dist:.2f} drone_z={drone_z_l:.2f} box_z={box_z_l:.2f}"
                phase[eid] = PHASE_DONE
                record_fail(eid, "box_dropped_delivery", detail)
                continue

            if dist_wp[i].item() < WP_REACH_DIST:
                wi = wp_idx[eid].item() + 1
                if wi < n_delivery_wps[eid].item():
                    wp_idx[eid] = wi
                    current_goal[eid] = delivery_wps[eid, wi]
                else:
                    phase[eid] = PHASE_ARRIVED
                    results["full_success"] += 1

    # ---- ARRIVED (phase==4) — hover briefly then recycle ----
    arr_mask = (phase == PHASE_ARRIVED)
    if arr_mask.any():
        arr_ids = arr_mask.nonzero(as_tuple=False).view(-1)
        obs_23d = compute_obs_batch(env, current_goal, prev_action_4d, arr_ids, loaded=True, payload=payload_mass)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[arr_ids] = act
        action_8d[arr_ids, :3] = act[:, :3]
        action_8d[arr_ids, 6] = act[:, 3]
        # Recycle after 1 step
        for eid in arr_ids.tolist():
            phase[eid] = PHASE_DONE

    # ---- bypass_analytical per-env ----
    bypass = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)
    bypass[phase == PHASE_DOCK] = False
    bypass[phase == PHASE_CLIMB] = False
    bypass[phase == PHASE_SETTLE] = True  # no PD during settle
    env.bypass_analytical = bypass

    # ---- Step ----
    env_wrapped.step(action_8d)

    # ---- Post-step: contain guard for approach ----
    approach_eids = (phase == PHASE_APPROACH).nonzero(as_tuple=False).view(-1)
    if len(approach_eids) > 0:
        env.contain_hold_count[approach_eids] = 0

    # ---- Self-managed termination (approach/delivery only) ----
    active = (phase == PHASE_APPROACH) | (phase == PHASE_DELIVERY)
    if active.any():
        act_ids = active.nonzero(as_tuple=False).view(-1)
        local_p = pos_w[act_ids] - env.scene.env_origins[act_ids]
        R_t = quat_to_rot_matrix(env.robot.data.root_quat_w[act_ids])
        tilt_t = torch.acos(R_t[:, 2, 2].clamp(-1, 1)) * 57.3

        for i, eid in enumerate(act_ids.tolist()):
            reason = None
            if local_p[i, 2].item() < 0.10:
                reason = "too_low"
            elif torch.norm(local_p[i, :2]).item() > 15.0:
                reason = "too_far"
            elif tilt_t[i].item() > 70:
                reason = "too_tilted"
            if reason:
                p = phase[eid].item()
                phase[eid] = PHASE_DONE
                phase_name = 'approach' if p == PHASE_APPROACH else 'delivery'
                record_fail(eid, f"{reason}_{phase_name}")

    # ---- Recycle done envs ----
    done_mask = (phase == PHASE_DONE)
    if done_mask.any():
        done_ids = done_mask.nonzero(as_tuple=False).view(-1)
        setup_mission(done_ids)
        # Commit box position for recycled envs
        for eid in done_ids.tolist():
            origin = env.scene.env_origins[eid]
            box_pos = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
            obj_state = torch.zeros(1, 13, device=device)
            obj_state[0, :3] = box_pos
            obj_state[0, 3] = 1.0
            env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
            env.object_pos[eid] = box_pos

    # ---- Status ----
    if step % 3000 == 0:
        total = results["total"] + results["full_success"]
        n = max(total, 1)
        print(f"  [{step/150:5.0f}s] A={a_mask.sum().item():>2} D={d_mask.sum().item():>2} "
              f"C={c_mask.sum().item():>2} V={v_mask.sum().item():>2} | "
              f"missions={total}/{TARGET_MISSIONS} "
              f"fails={results['total']} "
              f"success={results['full_success']} ({100*results['full_success']/n:.0f}%)")

# ============================================================
# Results
# ============================================================
n = max(results["total"] + results["full_success"], 1)
print(f"\n{'='*60}")
print(f"  END-TO-END PARALLEL RESULTS ({NUM_ENVS} envs)")
print(f"{'='*60}")
n_completed = results["total"] + results["full_success"]
n_dock_phase_fail = sum(v for k, v in fail_reasons.items() if "dock" in k)
n_climb_fail = sum(v for k, v in fail_reasons.items() if "climb" in k)
n_delivery_fail = sum(v for k, v in fail_reasons.items() if "delivery" in k)
n_approach_fail = sum(v for k, v in fail_reasons.items() if "approach" in k)
n_dock_entered = n_completed - n_approach_fail
n_grasp_retries = results.get("grasp_retries", 0)

print(f"  Total missions:    {n_completed}")
print(f"  Full success:      {results['full_success']}/{n_completed} ({100*results['full_success']/max(n_completed,1):.1f}%)")
print()
n_dock_success = n_dock_entered - n_dock_phase_fail
n_climb_success = n_dock_success - n_climb_fail
print(f"  --- Phase success rates ---")
print(f"  Approach → Dock:   {n_dock_entered}/{n_completed} ({100*n_dock_entered/max(n_completed,1):.1f}%)")
print(f"  Dock → Climb:      {n_dock_success}/{n_dock_entered} ({100*n_dock_success/max(n_dock_entered,1):.1f}%)  (retries: {n_grasp_retries})")
print(f"  Climb → Delivery:  {n_climb_success}/{n_dock_success} ({100*n_climb_success/max(n_dock_success,1):.1f}%)")
print(f"  Delivery → Done:   {results['full_success']}/{max(n_climb_success,1)} ({100*results['full_success']/max(n_climb_success,1):.1f}%)")
print()
print(f"  --- Failure breakdown ---")
for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason:30s} {count:>3}/{n_completed} ({100*count/max(n_completed,1):.1f}%)")
print(f"{'='*60}")

env.close()
simulation_app.close()
