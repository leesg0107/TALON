"""End-to-end mission eval — parallel version of eval_mission.py.

EXACT same logic as eval_mission.py, but NUM_ENVS missions run simultaneously.
Each env: teleport drone → approach WPs → PD dock → delivery WPs.
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import gymnasium as gym

# ============================================================
# Config — SAME as eval_mission.py
# ============================================================
NUM_ENVS = 100
MAX_STEPS = 18000  # 120s at 150Hz
DOCK_XY_SWITCH = 0.50
WP_REACH_DIST = 0.30

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = NUM_ENVS
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 120.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.spawn_spread = 0.0
cfg.truncate_on_dock_success = False
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

# Goal marker
goal_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/GoalMarker",
    markers={"goal": sim_utils.SphereCfg(radius=0.15,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0), opacity=0.6))},
)
try:
    goal_marker = VisualizationMarkers(goal_marker_cfg)
except Exception:
    goal_marker = None

# ============================================================
# Load models — SAME as eval_mission.py
# ============================================================
from envs.waypoint_cfg import WaypointEnvCfg

class FakeEnv:
    def __init__(self, obs_dim, act_dim, n):
        self.observation_space = gym.spaces.Box(-float('inf'), float('inf'), (obs_dim,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,))
        self.num_envs = n

agent_s1 = build_waypoint_agent(env=FakeEnv(22, 4, NUM_ENVS), device=device, mode="flight",
                                checkpoint_path="logs/gripper_wp_flight_v3/best_agent.pt")
agent_s1.set_running_mode("eval")

agent_s4 = build_waypoint_agent(env=FakeEnv(23, 4, NUM_ENVS), device=device, mode="loaded",
                                checkpoint_path="logs/gripper_wp_loaded_v10/best_agent.pt")
agent_s4.set_running_mode("eval")

# ============================================================
# Obs helpers — SAME as eval_mission.py, but batched
# ============================================================
_dr = WaypointEnvCfg().domain_rand

def compute_waypoint_obs_batch(env, goal, prev_act, eids):
    """Batch version of compute_waypoint_obs."""
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
    vel_b = vel_b + _dr.vel_noise_std * torch.randn_like(vel_b)
    goal_b = goal_b + _dr.pos_noise_std * torch.randn_like(goal_b)
    return torch.cat([vel_b, ang_vel_b, R_flat, goal_b, prev_norm], dim=-1)

def compute_loaded_obs_batch(env, goal, prev_act, payload, eids):
    obs22 = compute_waypoint_obs_batch(env, goal, prev_act, eids)
    pe = payload[eids].unsqueeze(-1) + 0.02 * torch.randn(len(eids), 1, device=device)
    return torch.cat([obs22, pe], dim=-1)

# ============================================================
# SAME generate_waypoints as eval_mission.py
# ============================================================
def generate_waypoints(start, target_xy, target_z=2.0, num_wps=3):
    waypoints = []
    cruise_z = start[2]
    for i in range(1, num_wps + 1):
        t = i / (num_wps + 1)
        wp_x = start[0] + t * (target_xy[0] - start[0]) + (torch.rand(1).item() - 0.5) * 0.6
        wp_y = start[1] + t * (target_xy[1] - start[1]) + (torch.rand(1).item() - 0.5) * 0.6
        wp_z = cruise_z + (torch.rand(1).item() - 0.5) * 0.4
        wp_z = max(wp_z, 1.5)
        waypoints.append(torch.tensor([wp_x, wp_y, wp_z], device=device))
    waypoints.append(torch.tensor([target_xy[0], target_xy[1], target_z], device=device))
    return waypoints

# ============================================================
# Per-env state
# ============================================================
# phase: "approach"=0, "dock"=1, "delivery"=2, "arrived"=3, "done"=4
phase = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
current_goal = torch.zeros(NUM_ENVS, 3, device=device)
prev_action_4d = torch.zeros(NUM_ENVS, 4, device=device)
wp_idx = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
payload_mass = torch.full((NUM_ENVS,), 0.2, device=device)

# Per-env waypoint lists (max 4 per phase)
MAX_WPS = 4
approach_wps = torch.zeros(NUM_ENVS, MAX_WPS, 3, device=device)
delivery_wps = torch.zeros(NUM_ENVS, MAX_WPS, 3, device=device)
n_approach_wps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
n_delivery_wps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)

mission_success = torch.zeros(NUM_ENVS, dtype=torch.bool, device=device)

# Results
results = {"total": 0, "full_success": 0, "dock_success": 0,
           "approach_fail": 0, "dock_fail": 0, "delivery_fail": 0}
fail_reasons = {}  # reason → count

# ============================================================
# Setup one mission for given envs — SAME as eval_mission.py per-mission setup
# ============================================================
def setup_mission(eids):
    """Teleport drone, reset box, generate WPs — exactly like eval_mission.py."""
    for eid in eids.tolist():
        # Random start position 2~3m from box (like eval_mission STARTS)
        box_pos = env.object_pos[eid].clone()
        angle = torch.rand(1).item() * 2 * math.pi
        dist = 2.0 + torch.rand(1).item() * 1.0
        start = [
            box_pos[0].item() + dist * math.cos(angle),
            box_pos[1].item() + dist * math.sin(angle),
            env.scene.env_origins[eid, 2].item() + 3.0,
        ]

        # Teleport drone — SAME as eval_mission.py
        root_state = torch.zeros(1, 13, device=device)
        root_state[0, :3] = torch.tensor(start, device=device)
        root_state[0, 3] = 1.0
        env.robot.write_root_state_to_sim(root_state, torch.tensor([eid], device=device))

        # Reset box to pedestal — SAME as eval_mission.py
        obj_state = torch.zeros(1, 13, device=device)
        obj_state[0, :3] = box_pos
        obj_state[0, 3] = 1.0
        env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))

        # Reset env state — SAME as eval_mission.py
        env.contain_hold_count[eid] = 0
        env.step_count[eid] = 0
        if hasattr(env, '_stuck_count'):
            env._stuck_count[eid] = 0

        # Generate approach WPs — SAME generate_waypoints function
        bxy = [box_pos[0].item(), box_pos[1].item()]
        target_z = box_pos[2].item() + 0.3
        wps = generate_waypoints(start, bxy, target_z=target_z, num_wps=3)
        for j, wp in enumerate(wps):
            approach_wps[eid, j] = wp
        n_approach_wps[eid] = len(wps)

        # Init state
        phase[eid] = 0  # approach
        wp_idx[eid] = 0
        current_goal[eid] = approach_wps[eid, 0]
        prev_action_4d[eid] = 0
        mission_success[eid] = False

    # Clear reset_buf for teleported envs — SAME as eval_mission.py
    env.reset_buf[eids] = False
    env.reset_terminated[eids] = False
    env.reset_time_outs[eids] = False

    # Note: skip dummy steps during mid-run re-setup (would affect other envs)
    # Only do dummy steps on initial setup (all envs at once)

# ============================================================
# Init all missions
# ============================================================
obs, _ = env_wrapped.reset()
all_ids = torch.arange(NUM_ENVS, device=device)
setup_mission(all_ids)

# Initial dummy steps to settle physics (all envs, safe at start)
env.bypass_analytical = True
for _ in range(3):
    dummy = torch.zeros(NUM_ENVS, 8, device=device)
    env_wrapped.step(dummy)
    env.reset_buf[:] = False
    env.reset_terminated[:] = False
    env.reset_time_outs[:] = False

print(f"\n{'='*60}")
print(f"  END-TO-END PARALLEL EVAL ({NUM_ENVS} envs)")
print(f"  Stage 1: gripper_wp_flight_v3")
print(f"  Stage 4: gripper_wp_loaded_v10")
print(f"{'='*60}\n")

# ============================================================
# Main loop — SAME logic as eval_mission.py, per-env
# ============================================================
for step in range(MAX_STEPS):
    # Read current state — SAME as eval_mission.py line 280
    pos_w = env.robot.data.root_pos_w
    obj_pos = env.object_pos

    action_8d = torch.zeros(NUM_ENVS, 8, device=device)

    # ---- APPROACH (phase==0) — SAME as eval_mission.py "approach" block ----
    a_mask = (phase == 0)
    if a_mask.any():
        a_ids = a_mask.nonzero(as_tuple=False).view(-1)

        # Compute obs & action — SAME
        obs_22d = compute_waypoint_obs_batch(env, current_goal, prev_action_4d, a_ids)
        with torch.no_grad():
            act = agent_s1.act(obs_22d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[a_ids] = act
        action_8d[a_ids, :3] = act[:, :3]
        action_8d[a_ids, 6] = act[:, 3]

        # WP reach — SAME
        dist_wp = torch.norm(pos_w[a_ids] - current_goal[a_ids], dim=-1)
        for i, eid in enumerate(a_ids.tolist()):
            if dist_wp[i].item() < WP_REACH_DIST:
                wi = wp_idx[eid].item() + 1
                if wi < n_approach_wps[eid].item():
                    wp_idx[eid] = wi
                    current_goal[eid] = approach_wps[eid, wi]

        # Dock switch — SAME
        xy_to_box = torch.norm(pos_w[a_ids, :2] - obj_pos[a_ids, :2], dim=-1)
        z_above = pos_w[a_ids, 2] - obj_pos[a_ids, 2]
        for i, eid in enumerate(a_ids.tolist()):
            if xy_to_box[i].item() < DOCK_XY_SWITCH and z_above[i].item() < 1.0:
                phase[eid] = 1

    # ---- DOCK (phase==1) — SAME as eval_mission.py "dock" block ----
    d_mask = (phase == 1)
    if d_mask.any():
        d_ids = d_mask.nonzero(as_tuple=False).view(-1)
        action_8d[d_ids] = 0  # PD controls

        # Check gripper — SAME
        gripper_angles = env.robot.data.joint_pos[d_ids][:, env.plate_joint_ids[0]]
        for i, eid in enumerate(d_ids.tolist()):
            if gripper_angles[i].item() < 0.2:
                results["dock_success"] += 1

                # Update mass — SAME
                drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[eid]
                env.attitude_ctrl.mass[eid] = drone_mass + 0.2

                # Generate delivery WPs from CURRENT position — SAME as eval_mission.py
                cur = pos_w[eid].tolist()
                delivery_target = [
                    (torch.rand(1).item() - 0.5) * 8.0,
                    (torch.rand(1).item() - 0.5) * 8.0,
                ]
                del_wps = generate_waypoints(cur, delivery_target, target_z=2.5, num_wps=3)
                for j, wp in enumerate(del_wps):
                    delivery_wps[eid, j] = wp
                n_delivery_wps[eid] = len(del_wps)

                phase[eid] = 2
                wp_idx[eid] = 0
                current_goal[eid] = delivery_wps[eid, 0]
                prev_action_4d[eid] = 0

    # ---- DELIVERY (phase==2) — SAME as eval_mission.py "delivery" block ----
    v_mask = (phase == 2)
    if v_mask.any():
        v_ids = v_mask.nonzero(as_tuple=False).view(-1)
        obs_23d = compute_loaded_obs_batch(env, current_goal, prev_action_4d, payload_mass, v_ids)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[v_ids] = act
        action_8d[v_ids, :3] = act[:, :3]
        action_8d[v_ids, 6] = act[:, 3]

        dist_wp = torch.norm(pos_w[v_ids] - current_goal[v_ids], dim=-1)
        for i, eid in enumerate(v_ids.tolist()):
            if dist_wp[i].item() < WP_REACH_DIST:
                wi = wp_idx[eid].item() + 1
                if wi < n_delivery_wps[eid].item():
                    wp_idx[eid] = wi
                    current_goal[eid] = delivery_wps[eid, wi]
                else:
                    phase[eid] = 3  # arrived
                    mission_success[eid] = True
                    results["full_success"] += 1

    # ---- ARRIVED (phase==3) — SAME as eval_mission.py "arrived" block ----
    arr_mask = (phase == 3)
    if arr_mask.any():
        arr_ids = arr_mask.nonzero(as_tuple=False).view(-1)
        obs_23d = compute_loaded_obs_batch(env, current_goal, prev_action_4d, payload_mass, arr_ids)
        with torch.no_grad():
            act = agent_s4.act(obs_23d, timestep=step, timesteps=MAX_STEPS)[0]
        prev_action_4d[arr_ids] = act
        action_8d[arr_ids, :3] = act[:, :3]
        action_8d[arr_ids, 6] = act[:, 3]

    # ---- bypass_analytical per-env ----
    bypass = torch.ones(NUM_ENVS, dtype=torch.bool, device=device)
    bypass[phase == 1] = False
    env.bypass_analytical = bypass

    # Visualize goals
    if goal_marker is not None:
        goal_marker.visualize(current_goal[:NUM_ENVS])

    # ---- Save pre-step state for failure diagnosis ----
    pre_pos_w = env.robot.data.root_pos_w.clone()
    pre_quat_w = env.robot.data.root_quat_w.clone()
    pre_obj_z = env.object_pos[:, 2].clone()

    # ---- Step — SAME as eval_mission.py ----
    obs, reward, terminated, truncated, info = env_wrapped.step(action_8d)

    # ---- Handle termination — SAME as eval_mission.py break logic, but per-env ----
    done = (terminated.view(-1) | truncated.view(-1))
    if done.any():
        done_ids = done.nonzero(as_tuple=False).view(-1)
        for eid in done_ids.tolist():
            p = phase[eid].item()
            if p < 3:  # not arrived = fail
                results["total"] += 1
                if p == 0: results["approach_fail"] += 1
                elif p == 1: results["dock_fail"] += 1
                elif p == 2: results["delivery_fail"] += 1

                # Diagnose WHY (using pre-step state)
                local_p = pre_pos_w[eid] - env.scene.env_origins[eid]
                R_d = quat_to_rot_matrix(pre_quat_w[eid:eid+1])
                tilt = torch.acos(R_d[0, 2, 2].clamp(-1, 1)).item() * 57.3
                box_z = pre_obj_z[eid].item() - env.scene.env_origins[eid, 2].item()
                is_term = terminated.view(-1)[eid].item()
                is_trunc = truncated.view(-1)[eid].item()

                if is_trunc and not is_term:
                    reason = "timeout"
                elif local_p[2].item() < 0.10:
                    reason = "too_low"
                elif torch.norm(local_p[:2]).item() > 10:
                    reason = "too_far"
                elif tilt > 60:
                    reason = "too_tilted"
                elif box_z < 0.30 and env.contain_hold_count[eid].item() < 150:
                    reason = "box_fell"
                else:
                    reason = "other"
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

            phase[eid] = 4  # done, waiting for re-setup

        # Re-setup failed/completed missions
        redo_ids = done_ids
        setup_mission(redo_ids)

    # Also re-setup arrived envs after 3s hover
    arrived_done = (phase == 3)
    # (let them hover, count is already done)

    # ---- Status ----
    if step % 1500 == 0:
        n_total = max(results["total"] + results["full_success"], 1)
        print(f"  [{step/150:5.0f}s] approach={a_mask.sum().item():>3} "
              f"dock={d_mask.sum().item():>3} "
              f"delivery={v_mask.sum().item():>3} "
              f"arrived={arr_mask.sum().item():>3} | "
              f"total={results['total']+results['full_success']} "
              f"dock_ok={results['dock_success']} "
              f"success={results['full_success']} "
              f"({100*results['full_success']/n_total:.0f}%)")

# ============================================================
# Results — SAME format
# ============================================================
n = max(results["total"] + results["full_success"], 1)
print(f"\n{'='*60}")
print(f"  END-TO-END MISSION RESULTS ({NUM_ENVS} parallel envs)")
print(f"{'='*60}")
print(f"  Total missions:    {results['total'] + results['full_success']}")
print(f"  Full success:      {results['full_success']}/{n} ({100*results['full_success']/n:.1f}%)")
print(f"  Dock success:      {results['dock_success']}/{n} ({100*results['dock_success']/n:.1f}%)")
print()
print(f"  --- Failure by phase ---")
print(f"  Approach fail:     {results['approach_fail']}/{n} ({100*results['approach_fail']/n:.1f}%)")
print(f"  Dock fail:         {results['dock_fail']}/{n} ({100*results['dock_fail']/n:.1f}%)")
print(f"  Delivery fail:     {results['delivery_fail']}/{n} ({100*results['delivery_fail']/n:.1f}%)")
print()
print(f"  --- Failure reasons ---")
for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"  {reason:>12}: {count}/{n} ({100*count/n:.1f}%)")
print(f"{'='*60}")

env.close()
simulation_app.close()
