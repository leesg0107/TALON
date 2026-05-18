"""Diagnostic eval for DELIVERY phase failures.

Same mission setup as eval_mission_headless.py, but during DELIVERY:
  - Per-step recording of (tilt, lateral_accel_cmd, box_offset_drift,
    box_relative_speed, steps_since_wp_change, wp_turn_angle).
  - On mission completion, classify outcome and snapshot delivery trajectory.

Post-run:
  - Aggregate success vs failure feature distributions (printed table).
  - Slip-onset → failure timing distribution.
  - Save raw trajectories to logs/diagnose_delivery_raw.npz for plotting.

Run:
    conda run -n isaaclab311 python scripts/diagnose_delivery.py
"""
import sys, os, math
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import gymnasium as gym

# ============================================================
# Config
# ============================================================
NUM_ENVS = 128
TARGET_MISSIONS = 500
MAX_STEPS = 18000
WP_REACH_DIST = 0.30
APPROACH_TIMEOUT = 9000
DOCK_TIMEOUT = 1800
MAX_DELIVERY_STEPS = 1500  # cap per-episode trace length (10s)
SLIP_THRESHOLD = 0.01      # m — box offset drift defining slip onset

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = NUM_ENVS
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 120.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.spawn_spread = 0.0
cfg.truncate_on_dock_success = False
cfg.goal_pos_range_xy = 0.0
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

def _eval_get_dones():
    return (torch.zeros(NUM_ENVS, dtype=torch.bool, device=device),
            torch.zeros(NUM_ENVS, dtype=torch.bool, device=device))
env._get_dones = _eval_get_dones

# ============================================================
# Load models
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
# Obs helpers
# ============================================================
_VEL_NOISE = 0.03
_POS_NOISE = 0.01

def compute_obs_batch(goal, prev_act, eids, loaded=False, payload=None):
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

def generate_waypoints(start, target_xy, target_z=2.0, num_wps=3, min_z=0.8):
    waypoints = []
    prev = list(start)
    for i in range(num_wps):
        dx = target_xy[0] - prev[0]; dy = target_xy[1] - prev[1]; dz = target_z - prev[2]
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
PHASE_SETTLE, PHASE_APPROACH, PHASE_DOCK, PHASE_CLIMB, PHASE_HOVER_STAB, PHASE_DELIVERY, PHASE_ARRIVED, PHASE_DONE = 0,1,2,3,4,5,6,7

phase = torch.full((NUM_ENVS,), PHASE_DONE, dtype=torch.long, device=device)
current_goal = torch.zeros(NUM_ENVS, 3, device=device)
prev_action_4d = torch.zeros(NUM_ENVS, 4, device=device)
wp_idx = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
payload_mass = torch.full((NUM_ENVS,), 0.2, device=device)
approach_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
post_wp_wait = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
dock_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
climb_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)

MAX_WPS = 5
approach_wps = torch.zeros(NUM_ENVS, MAX_WPS, 3, device=device)
delivery_wps = torch.zeros(NUM_ENVS, MAX_WPS, 3, device=device)
n_approach_wps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
n_delivery_wps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
delivery_target = torch.zeros(NUM_ENVS, 2, device=device)

# ---- Delivery diagnostic buffers (per env) ----
# Each env: a rolling buffer of delivery-phase per-step features.
# Features per step: [tilt_deg, lat_accel_norm, vert_accel_cmd, box_drift_norm,
#                     box_rel_speed, steps_since_wp, wp_turn_angle_deg, wp_idx]
N_FEAT = 8
delivery_traces = [None] * NUM_ENVS  # populated on DELIVERY entry
delivery_step_idx = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
delivery_initial_box_offset_b = torch.zeros(NUM_ENVS, 3, device=device)
delivery_wp_change_step = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
delivery_wp_turn_angle = torch.zeros(NUM_ENVS, device=device)
delivery_prev_wp_pos = torch.zeros(NUM_ENVS, 3, device=device)

# Aggregated results
all_trajectories = []   # list of (outcome:str, trace:np.array Nx8)
mission_records = []    # per-mission summary dict
results = {"total": 0, "full_success": 0}
fail_reasons = {}

def setup_mission(eids):
    for eid in eids.tolist():
        origin = env.scene.env_origins[eid]
        box_pos = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
        obj_state = torch.zeros(1, 13, device=device)
        obj_state[0, :3] = box_pos; obj_state[0, 3] = 1.0
        env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
        env.object_pos[eid] = box_pos

        angle = torch.rand(1).item() * 2 * math.pi
        dist = 2.0 + torch.rand(1).item() * 1.0
        start = [box_pos[0].item() + dist * math.cos(angle),
                 box_pos[1].item() + dist * math.sin(angle),
                 origin[2].item() + 3.0]

        root_state = torch.zeros(1, 13, device=device)
        root_state[0, :3] = torch.tensor(start, device=device); root_state[0, 3] = 1.0
        env.robot.write_root_state_to_sim(root_state, torch.tensor([eid], device=device))

        env.contain_hold_count[eid] = 0
        env.step_count[eid] = 0
        if hasattr(env, '_stuck_count'): env._stuck_count[eid] = 0
        if hasattr(env, '_recovery_timer'): env._recovery_timer[eid] = 0
        if hasattr(env, '_z_integral'): env._z_integral[eid] = 0.0
        if hasattr(env, '_dock_hold_z'): env._dock_hold_z[eid] = 0.0
        env.attitude_ctrl.reset(torch.tensor([eid], device=device))

        bxy = [box_pos[0].item(), box_pos[1].item()]
        target_z = box_pos[2].item() + 0.5
        wps = generate_waypoints(start, bxy, target_z=target_z, num_wps=3)
        for j, wp in enumerate(wps):
            approach_wps[eid, j] = wp
        n_approach_wps[eid] = len(wps)

        delivery_target[eid, 0] = origin[0].item() + (torch.rand(1).item() - 0.5) * 6.0
        delivery_target[eid, 1] = origin[1].item() + (torch.rand(1).item() - 0.5) * 6.0

        phase[eid] = PHASE_SETTLE
        wp_idx[eid] = 0
        current_goal[eid] = approach_wps[eid, 0]
        prev_action_4d[eid] = 0
        approach_steps[eid] = 0
        post_wp_wait[eid] = 0
        dock_steps[eid] = 0
        climb_steps[eid] = 0
        delivery_step_idx[eid] = 0
        delivery_traces[eid] = None

    env.reset_buf[eids] = False
    env.reset_terminated[eids] = False
    env.reset_time_outs[eids] = False

def finalize_mission(eid, outcome):
    """Store the delivery trace + summary for mission ending in this env."""
    results["total"] += 1
    if outcome == "full_success":
        results["full_success"] += 1
    else:
        fail_reasons[outcome] = fail_reasons.get(outcome, 0) + 1

    trace = delivery_traces[eid]
    if trace is not None:
        n = delivery_step_idx[eid].item()
        if n > 0:
            trace_arr = trace[:n].cpu().numpy()
            all_trajectories.append((outcome, trace_arr))

            # Summary features (per mission)
            max_tilt = float(trace_arr[:, 0].max())
            mean_tilt = float(trace_arr[:, 0].mean())
            max_lat_accel = float(trace_arr[:, 1].max())
            mean_lat_accel = float(trace_arr[:, 1].mean())
            max_drift = float(trace_arr[:, 3].max())
            max_rel_speed = float(trace_arr[:, 4].max())

            # Slip onset: first step where box_drift_norm > SLIP_THRESHOLD
            drift = trace_arr[:, 3]
            slip_idx = int(np.argmax(drift > SLIP_THRESHOLD)) if (drift > SLIP_THRESHOLD).any() else -1
            recovery_window = (n - slip_idx) if slip_idx >= 0 else -1

            # Failed near WP turn? steps_since_wp at failure
            steps_since_wp_at_end = float(trace_arr[-1, 5])
            wp_turn_at_fail = float(trace_arr[-1, 6])

            mission_records.append(dict(
                outcome=outcome,
                n_steps=n,
                max_tilt=max_tilt, mean_tilt=mean_tilt,
                max_lat_accel=max_lat_accel, mean_lat_accel=mean_lat_accel,
                max_box_drift=max_drift, max_rel_speed=max_rel_speed,
                slip_onset_step=slip_idx,
                recovery_window=recovery_window,
                steps_since_wp_at_end=steps_since_wp_at_end,
                wp_turn_at_fail=wp_turn_at_fail,
            ))

# ============================================================
# Init
# ============================================================
obs, _ = env_wrapped.reset()
all_ids = torch.arange(NUM_ENVS, device=device)
setup_mission(all_ids)

env.bypass_analytical = True
for _ in range(3):
    env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))
    env.contain_hold_count[:] = 0

for eid in range(NUM_ENVS):
    origin = env.scene.env_origins[eid]
    bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = bp; obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
    env.object_pos[eid] = bp

env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))
env.contain_hold_count[:] = 0
phase[:] = PHASE_APPROACH

print(f"\n{'='*60}")
print(f"  DELIVERY DIAGNOSTIC ({NUM_ENVS} envs, target={TARGET_MISSIONS} missions)")
print(f"{'='*60}\n")

# ============================================================
# Main loop
# ============================================================
for step in range(MAX_STEPS * 10):
    if results["total"] >= TARGET_MISSIONS:
        break

    pos_w = env.robot.data.root_pos_w
    vel_w = env.robot.data.root_lin_vel_w
    obj_pos = env.object_pos
    action_8d = torch.zeros(NUM_ENVS, 8, device=device)

    # ---- SETTLE ----
    s_mask = (phase == PHASE_SETTLE)
    if s_mask.any():
        s_ids = s_mask.nonzero(as_tuple=False).view(-1)
        for eid in s_ids.tolist():
            origin = env.scene.env_origins[eid]
            bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
            obj_st = torch.zeros(1, 13, device=device); obj_st[0, :3] = bp; obj_st[0, 3] = 1.0
            env.grasp_object.write_root_state_to_sim(obj_st, torch.tensor([eid], device=device))
            env.object_pos[eid] = bp
            env.contain_hold_count[eid] = 0
        phase[s_ids] = PHASE_APPROACH

    # ---- APPROACH ----
    a_mask = (phase == PHASE_APPROACH)
    if a_mask.any():
        a_ids = a_mask.nonzero(as_tuple=False).view(-1)
        approach_steps[a_ids] += 1
        obs_22d = compute_obs_batch(current_goal, prev_action_4d, a_ids)
        with torch.no_grad():
            act = agent_s1.act(obs_22d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[a_ids] = act
        action_8d[a_ids, :3] = act[:, :3]
        action_8d[a_ids, 6] = act[:, 3]

        dist_wp = torch.norm(pos_w[a_ids] - current_goal[a_ids], dim=-1)
        for i, eid in enumerate(a_ids.tolist()):
            if dist_wp[i].item() < WP_REACH_DIST:
                wi = wp_idx[eid].item() + 1
                if wi < n_approach_wps[eid].item():
                    wp_idx[eid] = wi
                    current_goal[eid] = approach_wps[eid, wi]

        xy_box = torch.norm(pos_w[a_ids, :2] - obj_pos[a_ids, :2], dim=-1)
        z_above = pos_w[a_ids, 2] - obj_pos[a_ids, 2]
        speed = torch.norm(vel_w[a_ids], dim=-1)
        R_a = quat_to_rot_matrix(env.robot.data.root_quat_w[a_ids])
        tilt = torch.acos(R_a[:, 2, 2].clamp(-1, 1)) * 57.3

        for i, eid in enumerate(a_ids.tolist()):
            last_wp = wp_idx[eid].item() >= n_approach_wps[eid].item()
            timeout = approach_steps[eid].item() > APPROACH_TIMEOUT
            if last_wp: post_wp_wait[eid] += 1
            if last_wp or timeout:
                pw = post_wp_wait[eid].item()
                sp = speed[i].item(); ti = tilt[i].item()
                xy = xy_box[i].item(); za = z_above[i].item()
                stable = sp < 2.0 and ti < 30 and xy < 0.8
                if pw > 750: stable = xy < 1.5 and za > 0 and za < 2.0
                if stable:
                    phase[eid] = PHASE_DOCK; dock_steps[eid] = 0
                elif timeout:
                    if xy < 2.0 and za > 0:
                        phase[eid] = PHASE_DOCK; dock_steps[eid] = 0
                    else:
                        phase[eid] = PHASE_DONE
                        finalize_mission(eid, "approach_timeout_too_far")
                else:
                    current_goal[eid] = approach_wps[eid, n_approach_wps[eid].item() - 1]

    # ---- DOCK ----
    d_mask = (phase == PHASE_DOCK)
    if d_mask.any():
        d_ids = d_mask.nonzero(as_tuple=False).view(-1)
        action_8d[d_ids] = 0
        dock_steps[d_ids] += 1
        for i, eid in enumerate(d_ids.tolist()):
            contain = env.contain_hold_count[eid].item()
            box_z_local = obj_pos[eid, 2].item() - env.scene.env_origins[eid, 2].item()
            if box_z_local < 0.30 and contain < 150:
                phase[eid] = PHASE_DONE
                finalize_mission(eid, "box_fell_during_dock"); continue
            if contain >= 325:
                box_drone_dist = torch.norm(pos_w[eid] - obj_pos[eid]).item()
                if box_drone_dist < 0.30:
                    drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[eid]
                    env.attitude_ctrl.mass[eid] = drone_mass + 0.2
                    phase[eid] = PHASE_CLIMB; climb_steps[eid] = 0
                else:
                    env.contain_hold_count[eid] = 0
                    if hasattr(env, '_stuck_count'): env._stuck_count[eid] = 0
                continue
            if dock_steps[eid].item() > DOCK_TIMEOUT:
                phase[eid] = PHASE_DONE
                finalize_mission(eid, "dock_timeout")

    # ---- CLIMB ----
    c_mask = (phase == PHASE_CLIMB)
    if c_mask.any():
        c_ids = c_mask.nonzero(as_tuple=False).view(-1)
        action_8d[c_ids] = 0
        climb_steps[c_ids] += 1
        for i, eid in enumerate(c_ids.tolist()):
            cur_z = pos_w[eid, 2].item(); box_z = obj_pos[eid, 2].item()
            box_z_local = box_z - env.scene.env_origins[eid, 2].item()
            bd = torch.norm(pos_w[eid] - obj_pos[eid]).item()
            cs = climb_steps[eid].item()
            if cs > 30 and box_z_local < 0.60 and cur_z - box_z > 0.25:
                env.contain_hold_count[eid] = 0
                if hasattr(env, '_stuck_count'): env._stuck_count[eid] = 0
                phase[eid] = PHASE_DOCK; dock_steps[eid] = 0; continue
            drone_z_local = cur_z - env.scene.env_origins[eid, 2].item()
            if cs > 60 and drone_z_local < 0.30:
                phase[eid] = PHASE_DONE
                finalize_mission(eid, "climb_failed"); continue
            if cur_z > 1.0:
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
                # ---- Initialize delivery trace ----
                delivery_traces[eid] = torch.zeros(MAX_DELIVERY_STEPS, N_FEAT, device=device)
                delivery_step_idx[eid] = 0
                # Initial box offset in body frame (gripper-centric)
                R_init = quat_to_rot_matrix(env.robot.data.root_quat_w[eid:eid+1])[0]
                gripper_pos_w_init = pos_w[eid] + R_init @ torch.tensor([0., 0., -0.08], device=device)
                box_rel_w = obj_pos[eid] - gripper_pos_w_init
                delivery_initial_box_offset_b[eid] = R_init.T @ box_rel_w
                delivery_wp_change_step[eid] = 0
                delivery_wp_turn_angle[eid] = 0.0
                delivery_prev_wp_pos[eid] = current_goal[eid].clone()
            elif cs > 1500:
                phase[eid] = PHASE_DONE; finalize_mission(eid, "climb_timeout")
            elif cur_z < 0.05:
                phase[eid] = PHASE_DONE; finalize_mission(eid, "climb_crashed")
            elif bd > 0.50:
                phase[eid] = PHASE_DONE; finalize_mission(eid, "climb_box_lost")

    # ---- DELIVERY (with per-step recording) ----
    v_mask = (phase == PHASE_DELIVERY)
    if v_mask.any():
        v_ids = v_mask.nonzero(as_tuple=False).view(-1)
        obs_23d = compute_obs_batch(current_goal, prev_action_4d, v_ids, loaded=True, payload=payload_mass)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[v_ids] = act
        action_8d[v_ids, :3] = act[:, :3]
        action_8d[v_ids, 6] = act[:, 3]

        # ---- Record per-step features ----
        # Scale action to physical units (matches env._scale_action)
        # action_low = [-8,-8,-8,-pi], action_high = [8,8,8,pi]
        accel_cmd = act[:, :3] * 8.0  # body-frame accel cmd [m/s²]
        R_v = quat_to_rot_matrix(env.robot.data.root_quat_w[v_ids])
        tilt_deg = torch.acos(R_v[:, 2, 2].clamp(-1, 1)) * 57.2958
        lat_accel = torch.norm(accel_cmd[:, :2], dim=-1)   # |ax,ay| in body frame
        vert_accel = accel_cmd[:, 2]
        # Box drift = current_box_offset_b - initial_box_offset_b
        gripper_offset_w = torch.bmm(R_v, torch.tensor([0., 0., -0.08], device=device).expand(len(v_ids), 3).unsqueeze(-1)).squeeze(-1)
        gripper_pos_w_v = pos_w[v_ids] + gripper_offset_w
        box_rel_w = obj_pos[v_ids] - gripper_pos_w_v
        box_offset_b = torch.bmm(R_v.transpose(1, 2), box_rel_w.unsqueeze(-1)).squeeze(-1)
        box_drift = box_offset_b - delivery_initial_box_offset_b[v_ids]
        box_drift_norm = torch.norm(box_drift, dim=-1)
        # Box relative speed (world)
        box_vel_w = env.grasp_object.data.root_lin_vel_w[v_ids]
        drone_vel_w = vel_w[v_ids]
        rel_vel = box_vel_w - drone_vel_w
        rel_speed = torch.norm(rel_vel, dim=-1)

        for i, eid in enumerate(v_ids.tolist()):
            d_idx = delivery_step_idx[eid].item()
            if d_idx < MAX_DELIVERY_STEPS:
                steps_since_wp = step - delivery_wp_change_step[eid].item()
                feats = torch.tensor([
                    tilt_deg[i].item(),
                    lat_accel[i].item(),
                    vert_accel[i].item(),
                    box_drift_norm[i].item(),
                    rel_speed[i].item(),
                    float(steps_since_wp),
                    delivery_wp_turn_angle[eid].item(),
                    float(wp_idx[eid].item()),
                ], device=device)
                delivery_traces[eid][d_idx] = feats
                delivery_step_idx[eid] = d_idx + 1

        dist_wp = torch.norm(pos_w[v_ids] - current_goal[v_ids], dim=-1)
        for i, eid in enumerate(v_ids.tolist()):
            box_z_l = obj_pos[eid, 2].item() - env.scene.env_origins[eid, 2].item()
            drone_z_l = pos_w[eid, 2].item() - env.scene.env_origins[eid, 2].item()
            box_dist = torch.norm(pos_w[eid] - obj_pos[eid]).item()
            if box_z_l < 0.30 or box_dist > 0.50:
                phase[eid] = PHASE_DONE
                finalize_mission(eid, "box_dropped_delivery")
                continue
            if dist_wp[i].item() < WP_REACH_DIST:
                wi = wp_idx[eid].item() + 1
                if wi < n_delivery_wps[eid].item():
                    # ---- WP change: record turn angle ----
                    new_wp = delivery_wps[eid, wi]
                    prev_wp = delivery_prev_wp_pos[eid]
                    cur_pos = pos_w[eid]
                    v1 = cur_pos - prev_wp
                    v2 = new_wp - cur_pos
                    n1 = torch.norm(v1[:2]); n2 = torch.norm(v2[:2])
                    if n1 > 1e-3 and n2 > 1e-3:
                        cos_a = (v1[:2] @ v2[:2]) / (n1 * n2)
                        cos_a = cos_a.clamp(-1, 1)
                        turn_angle_deg = math.degrees(math.acos(cos_a.item()))
                    else:
                        turn_angle_deg = 0.0
                    delivery_wp_turn_angle[eid] = turn_angle_deg
                    delivery_wp_change_step[eid] = step
                    delivery_prev_wp_pos[eid] = new_wp.clone()
                    wp_idx[eid] = wi
                    current_goal[eid] = new_wp
                else:
                    phase[eid] = PHASE_ARRIVED
                    finalize_mission(eid, "full_success")

    # ---- ARRIVED ----
    arr_mask = (phase == PHASE_ARRIVED)
    if arr_mask.any():
        arr_ids = arr_mask.nonzero(as_tuple=False).view(-1)
        obs_23d = compute_obs_batch(current_goal, prev_action_4d, arr_ids, loaded=True, payload=payload_mass)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[arr_ids] = act
        action_8d[arr_ids, :3] = act[:, :3]
        action_8d[arr_ids, 6] = act[:, 3]
        for eid in arr_ids.tolist():
            phase[eid] = PHASE_DONE

    # ---- bypass_analytical per-env ----
    bypass = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)
    bypass[phase == PHASE_DOCK] = False
    bypass[phase == PHASE_CLIMB] = False
    bypass[phase == PHASE_SETTLE] = True
    env.bypass_analytical = bypass

    env_wrapped.step(action_8d)

    approach_eids = (phase == PHASE_APPROACH).nonzero(as_tuple=False).view(-1)
    if len(approach_eids) > 0:
        env.contain_hold_count[approach_eids] = 0

    active = (phase == PHASE_APPROACH) | (phase == PHASE_DELIVERY)
    if active.any():
        act_ids = active.nonzero(as_tuple=False).view(-1)
        local_p = pos_w[act_ids] - env.scene.env_origins[act_ids]
        R_t = quat_to_rot_matrix(env.robot.data.root_quat_w[act_ids])
        tilt_t = torch.acos(R_t[:, 2, 2].clamp(-1, 1)) * 57.3
        for i, eid in enumerate(act_ids.tolist()):
            reason = None
            if local_p[i, 2].item() < 0.10: reason = "too_low"
            elif torch.norm(local_p[i, :2]).item() > 15.0: reason = "too_far"
            elif tilt_t[i].item() > 70: reason = "too_tilted"
            if reason:
                p = phase[eid].item()
                phase[eid] = PHASE_DONE
                phase_name = 'approach' if p == PHASE_APPROACH else 'delivery'
                finalize_mission(eid, f"{reason}_{phase_name}")

    done_mask = (phase == PHASE_DONE)
    if done_mask.any():
        done_ids = done_mask.nonzero(as_tuple=False).view(-1)
        setup_mission(done_ids)
        for eid in done_ids.tolist():
            origin = env.scene.env_origins[eid]
            bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
            obj_state = torch.zeros(1, 13, device=device)
            obj_state[0, :3] = bp; obj_state[0, 3] = 1.0
            env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
            env.object_pos[eid] = bp

    if step % 3000 == 0:
        n = max(results["total"], 1)
        print(f"  [{step/150:5.0f}s] missions={results['total']}/{TARGET_MISSIONS} "
              f"success={results['full_success']} ({100*results['full_success']/n:.0f}%) "
              f"records={len(mission_records)}")

# ============================================================
# Aggregate & print
# ============================================================
import json
print(f"\n{'='*60}\n  RESULTS\n{'='*60}")
n = results["total"]
print(f"  Total missions: {n}")
print(f"  Full success:   {results['full_success']} ({100*results['full_success']/max(n,1):.1f}%)")
for reason, c in sorted(fail_reasons.items(), key=lambda x: -x[1]):
    print(f"    {reason:30s} {c:>3} ({100*c/max(n,1):.1f}%)")

# Filter records: only those that reached delivery
delivery_records = [r for r in mission_records if r["n_steps"] > 0]
success_recs = [r for r in delivery_records if r["outcome"] == "full_success"]
fail_recs = [r for r in delivery_records if "delivery" in r["outcome"]]

print(f"\n{'='*60}\n  DELIVERY PHASE FEATURE COMPARISON (success vs failure)\n{'='*60}")
print(f"  Reached delivery: {len(delivery_records)} | success: {len(success_recs)} | fail: {len(fail_recs)}\n")

def stats(arr):
    a = np.array(arr) if len(arr) else np.array([0.0])
    return f"mean={a.mean():6.2f}  med={np.median(a):6.2f}  p90={np.percentile(a,90):6.2f}  max={a.max():6.2f}"

features = [
    ("max_tilt (deg)",          "max_tilt"),
    ("mean_tilt (deg)",         "mean_tilt"),
    ("max_lat_accel (m/s²)",    "max_lat_accel"),
    ("mean_lat_accel (m/s²)",   "mean_lat_accel"),
    ("max_box_drift (m)",       "max_box_drift"),
    ("max_rel_speed (m/s)",     "max_rel_speed"),
]
for label, key in features:
    s_vals = [r[key] for r in success_recs]
    f_vals = [r[key] for r in fail_recs]
    print(f"  {label:24s}")
    print(f"    success: {stats(s_vals)}")
    print(f"    fail:    {stats(f_vals)}")

# Slip onset → failure timing
slipped_fail = [r for r in fail_recs if r["slip_onset_step"] >= 0]
print(f"\n  Slip onset → failure window (frames @150Hz):")
if slipped_fail:
    rw = [r["recovery_window"] for r in slipped_fail]
    print(f"    n_failures_with_slip_signal: {len(slipped_fail)}/{len(fail_recs)}")
    print(f"    {stats(rw)}  [seconds: mean={np.mean(rw)/150:.3f}s]")
else:
    print(f"    No failures with detectable slip onset (threshold={SLIP_THRESHOLD}m)")

# WP turn correlation
print(f"\n  WP turn angle at failure (deg, last WP transition):")
print(f"    success: {stats([r['wp_turn_at_fail'] for r in success_recs])}")
print(f"    fail:    {stats([r['wp_turn_at_fail'] for r in fail_recs])}")
print(f"  Steps since last WP at failure:")
print(f"    success: {stats([r['steps_since_wp_at_end'] for r in success_recs])}")
print(f"    fail:    {stats([r['steps_since_wp_at_end'] for r in fail_recs])}")

# Save raw
os.makedirs("logs", exist_ok=True)
out_path = "logs/diagnose_delivery_raw.npz"
np.savez_compressed(
    out_path,
    records=np.array([json.dumps(r) for r in mission_records]),
    traces=np.array([t for _, t in all_trajectories], dtype=object),
    outcomes=np.array([o for o, _ in all_trajectories]),
)
print(f"\n  Raw data saved: {out_path}")
print(f"{'='*60}")

env.close()
simulation_app.close()
