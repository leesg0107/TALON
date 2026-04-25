"""End-to-end mission: Stage 1 waypoint → PD dock → Stage 4 delivery.

Uses GripperDroneEnv (has box+pedestal+gripper+PD).
Stage 1/4 waypoint models get 22D/23D obs computed directly from physics.
4D action is natively applied to attitude controller (no 8D conversion).

Phase 1: Stage 1 model flies through waypoints to above box
Phase 2: PD controller docks with box
Phase 3: (future) Stage 4 model flies to delivery
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

# ============================================================
# Environment: GripperDroneEnv (has box, pedestal, gripper, PD)
# ============================================================
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 120.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.spawn_spread = 0.0
cfg.truncate_on_dock_success = False  # don't auto-reset after dock
cfg.goal_pos_range_xy = 0.0          # box always at pedestal center (no random XY)
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

# ============================================================
# CRITICAL: Disable auto-reset inside DirectRLEnv.step()
# ============================================================
# DirectRLEnv.step() calls _get_dones() every step. If terminated,
# it calls _reset_idx() which undoes our teleport and resets drone state.
# This causes spurious "approach fail" from box_fell (not in training env)
# and makes diagnostic output show POST-RESET state (useless for debugging).
#
# Solution: monkey-patch _get_dones to always return False.
# eval_mission manages its own termination checks with accurate diagnostics.
_original_get_dones = env._get_dones

def _eval_get_dones():
    return (
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )

env._get_dones = _eval_get_dones

# ============================================================
# Load Stage 1 waypoint model (22D obs, 4D action)
# ============================================================
# Build agent with correct obs/action dims (no need for actual WaypointDroneEnv)
import gymnasium as gym
obs_space = gym.spaces.Box(-float('inf'), float('inf'), (22,))
act_space = gym.spaces.Box(-float('inf'), float('inf'), (4,))  # must match training: NO clipping

class FakeEnv:
    """Minimal wrapper to provide obs/action space for agent building."""
    def __init__(self, obs_dim, act_dim):
        self.observation_space = obs_space
        self.action_space = act_space
        self.num_envs = 1

fake_env_s1 = FakeEnv(22, 4)
agent_s1 = build_waypoint_agent(env=fake_env_s1, device=device, mode="flight",
                                checkpoint_path="logs/gripper_wp_flight_v6/final_agent.pt")
agent_s1.set_running_mode("eval")

# ============================================================
# Load Stage 4 waypoint model (23D obs, 4D action)
# ============================================================
obs_space_s4 = gym.spaces.Box(-float('inf'), float('inf'), (23,))
act_space_s4 = gym.spaces.Box(-float('inf'), float('inf'), (4,))  # must match training: NO clipping

class FakeEnvS4:
    def __init__(self):
        self.observation_space = obs_space_s4
        self.action_space = act_space_s4
        self.num_envs = 1

fake_env_s4 = FakeEnvS4()
agent_s4 = build_waypoint_agent(env=fake_env_s4, device=device, mode="loaded",
                                checkpoint_path="logs/gripper_wp_loaded_v16/best_agent.pt")
agent_s4.set_running_mode("eval")

# ============================================================
# Goal marker
# ============================================================
goal_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/GoalMarker",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                opacity=0.6,
            ),
        ),
    },
)
goal_marker = VisualizationMarkers(goal_marker_cfg)


# ============================================================
# Helper: compute 22D obs from GripperDroneEnv physics state
# ============================================================
# Noise std from training env (GripperWaypointEnv uses hardcoded 0.03/0.01)
_OBS_VEL_NOISE_STD = 0.03
_OBS_POS_NOISE_STD = 0.01


def compute_waypoint_obs(env, goal_pos, prev_action_4d):
    """Compute 22D obs matching GripperWaypointEnv from GripperDroneEnv state."""
    vel_b = env.robot.data.root_lin_vel_b[0]
    ang_vel_b = env.robot.data.root_ang_vel_b[0]
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    pos_w = env.robot.data.root_pos_w[0]

    # Goal in body frame
    goal_err_w = goal_pos - pos_w
    goal_b = R[0].T @ goal_err_w

    # Rotation matrix flattened
    R_flat = R[0].reshape(9)

    # Previous action normalized (matching GripperWaypointEnv)
    prev_norm = prev_action_4d.clone()
    prev_norm[:3] /= 8.0
    prev_norm[3] /= math.pi

    # Noise (matching training: gripper_waypoint_env.py line 181-182)
    vel_b_noisy = vel_b + _OBS_VEL_NOISE_STD * torch.randn_like(vel_b)
    goal_b_noisy = goal_b + _OBS_POS_NOISE_STD * torch.randn_like(goal_b)

    obs = torch.cat([vel_b_noisy, ang_vel_b, R_flat, goal_b_noisy, prev_norm])
    return obs.unsqueeze(0)  # (1, 22)


def compute_loaded_obs(env, goal_pos, prev_action_4d, payload_mass=0.2):
    """Compute 23D obs for Stage 4 (22D + payload_est)."""
    obs_22d = compute_waypoint_obs(env, goal_pos, prev_action_4d)
    payload_est = torch.tensor([[payload_mass + 0.02 * torch.randn(1).item()]], device=device)
    return torch.cat([obs_22d, payload_est], dim=-1)  # (1, 23)



# ============================================================
# Mission
# ============================================================
NUM_MISSIONS = 300
DOCK_XY_SWITCH = 0.50
WP_REACH_DIST = 0.30

# Generate random starts (2~3m from origin at z=3.0)
import random
STARTS = []
for _ in range(NUM_MISSIONS):
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(2.0, 3.0)
    STARTS.append([dist * math.cos(angle), dist * math.sin(angle), 3.0])

def generate_waypoints(start, target_xy, target_z=2.0, num_wps=3, min_z=0.8):
    """Generate waypoints as random hops (matching training pattern).

    Each intermediate WP: random direction 0.8~2.3m from previous, random Z.
    Final WP: exact target position.
    """
    waypoints = []
    prev = list(start)
    for i in range(num_wps):
        # Random hop from previous WP, but bias toward target
        dx = target_xy[0] - prev[0]
        dy = target_xy[1] - prev[1]
        dz = target_z - prev[2]
        remaining = math.sqrt(dx**2 + dy**2)

        # Direction: blend between random and toward-target
        angle_to_target = math.atan2(dy, dx)
        angle = angle_to_target + (torch.rand(1).item() - 0.5) * 1.5  # ±0.75 rad wobble
        dist = 0.8 + torch.rand(1).item() * 1.5  # 0.8~2.3m (same as training)
        dist = min(dist, remaining + 0.5)  # don't overshoot

        wp_x = prev[0] + dist * math.cos(angle)
        wp_y = prev[1] + dist * math.sin(angle)
        # Z: random within flight range, biased toward target
        t = (i + 1) / (num_wps + 1)
        wp_z = prev[2] + t * dz * 0.5 + (torch.rand(1).item() - 0.5) * 1.0
        wp_z = max(wp_z, min_z)

        waypoints.append(torch.tensor([wp_x, wp_y, wp_z], device=device))
        prev = [wp_x, wp_y, wp_z]

    # Final WP: exact target
    waypoints.append(torch.tensor([target_xy[0], target_xy[1], target_z], device=device))
    return waypoints

print(f"\n{'='*60}")
print(f"  END-TO-END MISSION (Stage 1 + PD Dock + Stage 4)")
print(f"  Stage 1: logs/gripper_wp_flight_v6/final_agent.pt")
print(f"  PD dock: analytical controller")
print(f"  Stage 4: logs/gripper_wp_loaded_v16/best_agent.pt")
print(f"{'='*60}\n")

obs, _ = env_wrapped.reset()
mission_results = []

for mission in range(NUM_MISSIONS):
    # Fresh reset
    obs, _ = env_wrapped.reset()

    start_pos = torch.tensor(STARTS[mission], device=device)

    # Box position: center of pedestal (NOT from _sample_objects which randomizes XY)
    # Pedestal is 30x30cm at env origin, top at z=0.50, box half=0.04 → box at z=0.54
    env_origin = env.scene.env_origins[0]
    box_pos = torch.tensor([env_origin[0].item(), env_origin[1].item(),
                            env_origin[2].item() + 0.54], device=device)

    # Teleport drone to start position
    root_state = torch.zeros(1, 13, device=device)
    root_state[0, :3] = start_pos
    root_state[0, 3] = 1.0
    env.robot.write_root_state_to_sim(root_state)

    # Reset box to pedestal center (zero velocity)
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = box_pos
    obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([0], device=device))

    # Update env's cached object_pos to match
    env.object_pos[0] = box_pos

    # Reset env state
    env.contain_hold_count[0] = 0
    env.step_count[0] = 0
    if hasattr(env, '_stuck_count'):
        env._stuck_count[0] = 0
    env.attitude_ctrl.reset(torch.tensor([0], device=device))

    # CRITICAL: clear reset_buf so step() doesn't auto-reset and overwrite teleport
    env.reset_buf[:] = False
    env.reset_terminated[:] = False
    env.reset_time_outs[:] = False

    # Dummy steps to apply teleport to physics (bypass analytical for RL control)
    env.bypass_analytical = True
    for _ in range(3):  # few steps to settle physics
        dummy = torch.zeros(1, 8, device=device)
        env_wrapped.step(dummy)
        env.reset_buf[:] = False
        env.reset_terminated[:] = False
        env.reset_time_outs[:] = False

    # Re-place box AFTER dummy steps (physics may have moved it)
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = box_pos
    obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([0], device=device))
    env.object_pos[0] = box_pos
    # One more step to commit the box position
    env_wrapped.step(torch.zeros(1, 8, device=device))
    env.contain_hold_count[0] = 0

    # Generate approach waypoints: start → WP1 → WP2 → WP3 → 50cm above box
    approach_wps = generate_waypoints(STARTS[mission], [box_pos[0].item(), box_pos[1].item()],
                                       target_z=box_pos[2].item() + 0.5)
    wp_idx = 0
    current_goal = approach_wps[wp_idx]

    # Delivery target (random, far from box)
    delivery_target = [
        env_origin[0].item() + (torch.rand(1).item() - 0.5) * 6.0,  # ±3m from origin
        env_origin[1].item() + (torch.rand(1).item() - 0.5) * 6.0,
    ]
    delivery_wps = []  # generated after dock

    phase = "approach"
    prev_action_4d = torch.zeros(4, device=device)

    print(f"\n  --- Mission {mission+1}/{NUM_MISSIONS} ---")
    print(f"  Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
    print(f"  Approach waypoints:")
    for wi, wp in enumerate(approach_wps):
        tag = " (above box)" if wi == len(approach_wps)-1 else ""
        print(f"    WP{wi+1}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}){tag}")
    print(f"  Delivery target: ({delivery_target[0]:.1f}, {delivery_target[1]:.1f})")

    mission_success = False
    dock_time = 0

    dock_timeout_steps = 0
    DOCK_TIMEOUT = 1800      # 12s max for dock attempt
    APPROACH_TIMEOUT = 9000  # 60s max for approach
    approach_steps = 0
    post_wp_wait = 0         # steps since last WP reached (for relaxed transition)
    fail_reason = None

    for step in range(18000):  # 120s max
        pos_w = env.robot.data.root_pos_w[0]
        obj_pos = env.object_pos[0]
        xy_to_box = torch.norm(pos_w[:2] - obj_pos[:2]).item()
        z_above_box = pos_w[2].item() - obj_pos[2].item()
        box_z = obj_pos[2].item()

        # ============ APPROACH ============
        if phase == "approach":
            approach_steps += 1
            env.bypass_analytical = True
            obs_22d = compute_waypoint_obs(env, current_goal, prev_action_4d)
            with torch.no_grad():
                action_4d_raw = agent_s1.act(obs_22d, timestep=step, timesteps=18000)[0][0]
            prev_action_4d = action_4d_raw.clone()

            goal_marker.visualize(current_goal.unsqueeze(0))

            dist_to_wp = torch.norm(pos_w - current_goal).item()
            if dist_to_wp < WP_REACH_DIST:
                wp_idx += 1
                if wp_idx < len(approach_wps):
                    current_goal = approach_wps[wp_idx]
                    tag = " (above box)" if wp_idx == len(approach_wps)-1 else ""
                    print(f"    [{step/150:.1f}s] WP{wp_idx} reached → WP{wp_idx+1}{tag}")

            # Transition: last WP reached → dock (relaxed stability check)
            last_wp_done = wp_idx >= len(approach_wps)
            approach_timeout = approach_steps > APPROACH_TIMEOUT

            if last_wp_done:
                post_wp_wait += 1

            if last_wp_done or approach_timeout:
                vel = env.robot.data.root_lin_vel_w[0]
                speed = torch.norm(vel).item()
                R_check = quat_to_rot_matrix(env.robot.data.root_quat_w)
                tilt_deg = torch.acos(R_check[0, 2, 2].clamp(-1, 1)).item() * 57.3

                stable = speed < 2.0 and tilt_deg < 30 and xy_to_box < 0.8
                if post_wp_wait > 750:
                    stable = xy_to_box < 1.5 and z_above_box > 0 and z_above_box < 2.0

                if stable:
                    phase = "dock"
                    dock_timeout_steps = 0
                    env.bypass_analytical = False
                    reason = f"stable (v={speed:.1f} tilt={tilt_deg:.0f}° wait={post_wp_wait/150:.1f}s)"
                    print(f"    [{step/150:.1f}s] APPROACH → DOCK ({reason}, xy={xy_to_box:.2f}m z_above={z_above_box:.2f}m)")
                elif approach_timeout:
                    if xy_to_box < 2.0 and z_above_box > 0:
                        phase = "dock"
                        dock_timeout_steps = 0
                        env.bypass_analytical = False
                        print(f"    [{step/150:.1f}s] APPROACH → DOCK (TIMEOUT, xy={xy_to_box:.2f}m z_above={z_above_box:.2f}m)")
                    else:
                        fail_reason = "approach_timeout_too_far"
                        print(f"    [{step/150:.1f}s] APPROACH TIMEOUT — too far (xy={xy_to_box:.2f}m z_above={z_above_box:.2f}m)")
                        break
                else:
                    # Keep flying toward last WP
                    current_goal = approach_wps[-1]

        # ============ DOCK ============
        elif phase == "dock":
            env.bypass_analytical = False
            dock_timeout_steps += 1
            goal_marker.visualize(torch.tensor([[0, 0, -10]], device=device, dtype=torch.float))

            contain = env.contain_hold_count[0].item()
            gripper_angle = env.robot.data.joint_pos[0, env.plate_joint_ids[0]].item()

            if step % 150 == 0:
                print(f"      contain={contain}/150 gripper={gripper_angle:.3f} "
                      f"xy={xy_to_box:.2f} z_above={z_above_box:.2f} "
                      f"box_z={box_z:.2f} dock_t={dock_timeout_steps/150:.1f}s")

            # Check box fell off pedestal (local z < 0.30 means it's on the ground)
            box_z_local = box_z - env.scene.env_origins[0, 2].item()
            if box_z_local < 0.30 and contain < 150:
                fail_reason = "box_fell_during_dock"
                print(f"    [{step/150:.1f}s] BOX FELL (box_z_local={box_z_local:.2f}) — dock abort")
                break

            if contain >= 200:
                box_drone_dist = torch.norm(pos_w - obj_pos).item()

                if box_drone_dist < 0.30:
                    # === BOX GRASPED (box following drone) → climb ===
                    dock_time = step / 150.0
                    drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[0]
                    env.attitude_ctrl.mass[0] = drone_mass + 0.2

                    print(f"    [{step/150:.1f}s] *** GRIP CONFIRMED *** (contain={contain}, box_dist={box_drone_dist:.3f})")
                    climb_wait = 0
                    climb_ok = False
                    while climb_wait < 900:
                        env_wrapped.step(torch.zeros(1, 8, device=device))
                        climb_wait += 1
                        cur_z = env.robot.data.root_pos_w[0, 2].item()
                        if cur_z > 1.0:
                            climb_ok = True
                            break
                        _bd = torch.norm(env.robot.data.root_pos_w[0] - env.object_pos[0]).item()
                        if _bd > 0.50 or cur_z < 0.05:
                            break
                    step += climb_wait

                    if not climb_ok:
                        fail_reason = "climb_failed"
                        print(f"    [{step/150:.1f}s] CLIMB FAILED (z={cur_z:.2f}) after {climb_wait} steps")
                        break

                    cur_pos = env.robot.data.root_pos_w[0].tolist()
                    delivery_wps = generate_waypoints(
                        cur_pos, delivery_target, target_z=2.5, num_wps=3, min_z=1.5
                    )
                    wp_idx = 0
                    current_goal = delivery_wps[wp_idx]
                    prev_action_4d = torch.zeros(4, device=device)

                    phase = "delivery"
                    env.bypass_analytical = True
                    print(f"    [{step/150:.1f}s] *** DOCKED *** ({dock_time:.1f}s, climb_z={cur_pos[2]:.2f}) → DELIVERY")
                    for wi, wp in enumerate(delivery_wps):
                        tag = " (final)" if wi == len(delivery_wps)-1 else ""
                        print(f"      WP{wi+1}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}){tag}")

                else:
                    # === GRIP MISSED → reset contain, gripper re-opens, PD retries ===
                    env.contain_hold_count[0] = 0
                    if hasattr(env, '_stuck_count'):
                        env._stuck_count[0] = 0
                    print(f"    [{step/150:.1f}s] GRIP MISSED (box_dist={box_drone_dist:.2f}) — retry")

            elif dock_timeout_steps > DOCK_TIMEOUT:
                fail_reason = "dock_timeout"
                _dt_xy = xy_to_box
                _dt_z = z_above_box
                _dt_spd = torch.norm(env.robot.data.root_lin_vel_w[0]).item()
                _dt_tilt = torch.acos(quat_to_rot_matrix(env.robot.data.root_quat_w)[0, 2, 2].clamp(-1, 1)).item() * 57.3
                if contain >= 150:
                    detail = f"gripper_closed_but_no_climb contain={contain}"
                elif contain > 50:
                    detail = f"partial_contain={contain}"
                elif _dt_xy > 0.15:
                    detail = f"xy_misaligned xy={_dt_xy:.2f}"
                elif _dt_z > 0.30:
                    detail = f"hovering_high z_above={_dt_z:.2f}"
                else:
                    detail = f"stuck xy={_dt_xy:.2f} z={_dt_z:.2f} v={_dt_spd:.2f} tilt={_dt_tilt:.0f}"
                print(f"    [{step/150:.1f}s] DOCK TIMEOUT: {detail} (contain={contain})")
                break

        # ============ DELIVERY ============
        elif phase == "delivery":
            env.bypass_analytical = True

            # Check box still held (box should be near drone, not on ground)
            box_drone_dist = torch.norm(pos_w - obj_pos).item()
            if box_z < 0.30 or box_drone_dist > 0.50:
                fail_reason = "box_dropped_during_delivery"
                print(f"    [{step/150:.1f}s] BOX DROPPED (box_z={box_z:.2f}, dist={box_drone_dist:.2f})")
                break

            obs_23d = compute_loaded_obs(env, current_goal, prev_action_4d, payload_mass=0.2)
            with torch.no_grad():
                action_4d_raw = agent_s4.act(obs_23d, timestep=step, timesteps=18000)[0][0]
            prev_action_4d = action_4d_raw.clone()

            goal_marker.visualize(current_goal.unsqueeze(0))

            dist_to_wp = torch.norm(pos_w - current_goal).item()
            if dist_to_wp < WP_REACH_DIST:
                wp_idx += 1
                if wp_idx < len(delivery_wps):
                    current_goal = delivery_wps[wp_idx]
                    tag = " (final)" if wp_idx == len(delivery_wps)-1 else ""
                    print(f"    [{step/150:.1f}s] Delivery WP{wp_idx} → WP{wp_idx+1}{tag}")
                else:
                    phase = "arrived"
                    mission_success = True
                    total_time = step / 150.0
                    print(f"    [{step/150:.1f}s] *** MISSION COMPLETE *** (total={total_time:.1f}s)")

        # ============ ARRIVED ============
        elif phase == "arrived":
            env.bypass_analytical = True
            obs_23d = compute_loaded_obs(env, current_goal, prev_action_4d, payload_mass=0.2)
            with torch.no_grad():
                action_4d_raw = agent_s4.act(obs_23d, timestep=step, timesteps=18000)[0][0]
            prev_action_4d = action_4d_raw.clone()
            goal_marker.visualize(torch.tensor([[0, 0, -10]], device=device, dtype=torch.float))

        # ============ STEP PHYSICS ============
        if phase == "dock":
            dummy_8d = torch.zeros(1, 8, device=device)
            env_wrapped.step(dummy_8d)
        elif phase != "arrived" or (phase == "arrived" and step <= total_time * 150 + 450):
            action_8d = torch.zeros(1, 8, device=device)
            action_8d[0, :3] = prev_action_4d[:3]
            action_8d[0, 6] = prev_action_4d[3]
            env_wrapped.step(action_8d)
        else:
            break

        # ============ POST-STEP: contain_hold_count guard ============
        # _get_rewards() increments contain_hold_count even during approach.
        # This can trigger gripper auto-close (lock_gripper threshold=150).
        # Reset it during approach so only dock phase accumulates.
        if phase == "approach":
            env.contain_hold_count[0] = 0

        # ============ SELF-MANAGED TERMINATION ============
        # We disabled env's _get_dones (no auto-reset), so check here.
        # Reads PRE-RESET state → diagnostics are accurate.
        _p = env.robot.data.root_pos_w[0]
        _local_p = _p - env.scene.env_origins[0]
        _R = quat_to_rot_matrix(env.robot.data.root_quat_w)
        _tilt_deg = torch.acos(_R[0, 2, 2].clamp(-1, 1)).item() * 57.3
        _box_z_local = env.object_pos[0, 2].item() - env.scene.env_origins[0, 2].item()

        _terminated = False
        if _local_p[2].item() < 0.10:
            fail_reason = "too_low"
            _terminated = True
        elif torch.norm(_local_p[:2]).item() > 10.0:
            fail_reason = "too_far"
            _terminated = True
        elif _tilt_deg > 60:
            fail_reason = "too_tilted"
            _terminated = True
        elif phase == "dock" and _box_z_local < 0.30 and env.contain_hold_count[0].item() < 150:
            fail_reason = "box_fell_during_dock"
            _terminated = True

        if _terminated:
            print(f"    [{step/150:.1f}s] TERMINATED: {fail_reason} at phase={phase}")
            print(f"      pos=({_p[0]:.1f},{_p[1]:.1f},{_p[2]:.1f}) tilt={_tilt_deg:.0f}°")
            print(f"      local_z={_local_p[2].item():.2f} xy_dist={torch.norm(_local_p[:2]).item():.1f}")
            print(f"      box_z_local={_box_z_local:.2f} contain={env.contain_hold_count[0].item()}")
            break

        # Status
        if step % 150 == 0:
            extra = ""
            if phase == "approach":
                extra = f"wp={wp_idx+1}/{len(approach_wps)} dist_wp={dist_to_wp:.2f}"
            elif phase == "dock":
                extra = f"contain={env.contain_hold_count[0].item()}/150"
            elif phase == "delivery":
                extra = f"wp={wp_idx+1}/{len(delivery_wps)} dist_wp={dist_to_wp:.2f}"
            print(f"    [{step/150:5.1f}s] {phase:>10}  pos=({pos_w[0]:.1f},{pos_w[1]:.1f},{pos_w[2]:.1f})  "
                  f"xy_box={xy_to_box:.2f}  {extra}")

        if phase == "arrived" and step > total_time * 150 + 450:
            break

    if not mission_success and fail_reason is None:
        fail_reason = f"terminated_in_{phase}"
    result = "SUCCESS" if mission_success else f"FAIL ({fail_reason or phase})"
    mission_results.append({"success": mission_success,
                            "fail_phase": phase if not mission_success else None,
                            "fail_reason": fail_reason,
                            "dock_time": dock_time if dock_time > 0 else None})
    print(f"  Result: {result}")

n_total = len(mission_results)
n_success = sum(1 for r in mission_results if r["success"])
n_approach_fail = sum(1 for r in mission_results if r["fail_phase"] == "approach")
n_dock_fail = sum(1 for r in mission_results if r["fail_phase"] == "dock")
n_delivery_fail = sum(1 for r in mission_results if r["fail_phase"] == "delivery")
n_docked = sum(1 for r in mission_results if r["dock_time"] is not None)

print(f"\n{'='*60}")
print(f"  MISSION SUMMARY")
print(f"{'='*60}")
print(f"  Total missions:    {n_total}")
print(f"  Full success:      {n_success}/{n_total} ({100*n_success/max(n_total,1):.1f}%)")
print(f"  Dock achieved:     {n_docked}/{n_total} ({100*n_docked/max(n_total,1):.1f}%)")
print(f"")
print(f"  --- Failure by phase ---")
print(f"  Approach fail:     {n_approach_fail}/{n_total} ({100*n_approach_fail/max(n_total,1):.1f}%)")
print(f"  Dock fail:         {n_dock_fail}/{n_total} ({100*n_dock_fail/max(n_total,1):.1f}%)")
print(f"  Delivery fail:     {n_delivery_fail}/{n_total} ({100*n_delivery_fail/max(n_total,1):.1f}%)")

# Failure reason breakdown
fail_reason_counts = {}
for r in mission_results:
    if r["fail_reason"]:
        fail_reason_counts[r["fail_reason"]] = fail_reason_counts.get(r["fail_reason"], 0) + 1
if fail_reason_counts:
    print(f"")
    print(f"  --- Failure reasons ---")
    for reason, count in sorted(fail_reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s} {count:>3}/{n_total} ({100*count/max(n_total,1):.1f}%)")
print(f"")
print(f"  --- Per mission ---")
for i, r in enumerate(mission_results):
    if r["success"]:
        print(f"    Mission {i+1}: SUCCESS (dock at {r['dock_time']:.1f}s)")
    elif r["fail_phase"] == "dock" and r["dock_time"]:
        print(f"    Mission {i+1}: FAIL (delivery) — docked at {r['dock_time']:.1f}s")
    else:
        print(f"    Mission {i+1}: FAIL ({r['fail_phase']})")
print(f"{'='*60}")

env.close()
simulation_app.close()
