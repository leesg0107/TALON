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
app_launcher = AppLauncher(headless=False)
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
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

# ============================================================
# Load Stage 1 waypoint model (22D obs, 4D action)
# ============================================================
# Create a dummy WaypointDroneEnv just for agent building (obs/action space)
from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv

dummy_cfg = WaypointEnvCfg(mode="flight")
dummy_cfg.scene.num_envs = 1
# We DON'T create the env, just use cfg for spaces
# Build agent with correct obs/action dims
import gymnasium as gym
obs_space = gym.spaces.Box(-float('inf'), float('inf'), (22,))
act_space = gym.spaces.Box(-1.0, 1.0, (4,))

class FakeEnv:
    """Minimal wrapper to provide obs/action space for agent building."""
    def __init__(self, obs_dim, act_dim):
        self.observation_space = obs_space
        self.action_space = act_space
        self.num_envs = 1

fake_env_s1 = FakeEnv(22, 4)
agent_s1 = build_waypoint_agent(env=fake_env_s1, device=device, mode="flight",
                                checkpoint_path="logs/gripper_wp_flight_v5/best_agent.pt")
agent_s1.set_running_mode("eval")

# ============================================================
# Load Stage 4 waypoint model (23D obs, 4D action)
# ============================================================
obs_space_s4 = gym.spaces.Box(-float('inf'), float('inf'), (23,))
act_space_s4 = gym.spaces.Box(-1.0, 1.0, (4,))

class FakeEnvS4:
    def __init__(self):
        self.observation_space = obs_space_s4
        self.action_space = act_space_s4
        self.num_envs = 1

fake_env_s4 = FakeEnvS4()
agent_s4 = build_waypoint_agent(env=fake_env_s4, device=device, mode="loaded",
                                checkpoint_path="logs/gripper_wp_loaded_v12/best_agent.pt")
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
def compute_waypoint_obs(env, goal_pos, prev_action_4d):
    """Compute 22D obs matching WaypointDroneEnv from GripperDroneEnv state."""
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

    # Previous action normalized (matching WaypointDroneEnv)
    prev_norm = prev_action_4d.clone()
    prev_norm[:3] /= 8.0
    prev_norm[3] /= math.pi

    # Noise (matching training)
    dr = WaypointEnvCfg().domain_rand
    vel_b_noisy = vel_b + dr.vel_noise_std * torch.randn_like(vel_b)
    goal_b_noisy = goal_b + dr.pos_noise_std * torch.randn_like(goal_b)

    obs = torch.cat([vel_b_noisy, ang_vel_b, R_flat, goal_b_noisy, prev_norm])
    return obs.unsqueeze(0)  # (1, 22)


def compute_loaded_obs(env, goal_pos, prev_action_4d, payload_mass=0.2):
    """Compute 23D obs for Stage 4 (22D + payload_est)."""
    obs_22d = compute_waypoint_obs(env, goal_pos, prev_action_4d)
    payload_est = torch.tensor([[payload_mass + 0.02 * torch.randn(1).item()]], device=device)
    return torch.cat([obs_22d, payload_est], dim=-1)  # (1, 23)


# ============================================================
# Helper: apply 4D action directly to attitude controller
# ============================================================
def apply_waypoint_action(env, action_4d_raw):
    """Scale 4D raw [-1,1] action and store for attitude controller.

    This sets the env's internal state so that _apply_action uses these values.
    """
    # Scale: same as WaypointDroneEnv
    accel_b = action_4d_raw[:3] * 8.0  # [-1,1] → [-8,8]
    yaw_ref = action_4d_raw[3] * math.pi  # [-1,1] → [-π,π]

    # Store in env's scaled_actions format (8D)
    env.scaled_actions = torch.zeros(1, 8, device=device)
    env.scaled_actions[0, :3] = accel_b
    env.scaled_actions[0, 6] = yaw_ref
    # [3:6] = 0 (rate_cmd), [7] = gripper (handled by lock_gripper)

    # Store raw for reward smoothness calc
    env.prev_action = env.raw_actions.clone()
    env.raw_actions = torch.zeros(1, 8, device=device)
    env.raw_actions[0, :3] = action_4d_raw[:3]
    env.raw_actions[0, 6] = action_4d_raw[3]

    return accel_b, yaw_ref


# ============================================================
# Mission
# ============================================================
NUM_MISSIONS = 30
DOCK_XY_SWITCH = 0.50
WP_REACH_DIST = 0.30

# Generate random starts (2~3m from origin at z=3.0)
import random
STARTS = []
for _ in range(NUM_MISSIONS):
    angle = random.uniform(0, 2 * math.pi)
    dist = random.uniform(2.0, 3.0)
    STARTS.append([dist * math.cos(angle), dist * math.sin(angle), 3.0])

def generate_waypoints(start, target_xy, target_z=2.0, num_wps=3):
    """Generate waypoints with gradual descent from start to target altitude."""
    waypoints = []
    for i in range(1, num_wps + 1):
        t = i / (num_wps + 1)
        wp_x = start[0] + t * (target_xy[0] - start[0]) + (torch.rand(1).item() - 0.5) * 0.6
        wp_y = start[1] + t * (target_xy[1] - start[1]) + (torch.rand(1).item() - 0.5) * 0.6
        # Gradual altitude interpolation from start to target
        wp_z = start[2] + t * (target_z - start[2]) + (torch.rand(1).item() - 0.5) * 0.4
        wp_z = max(wp_z, 0.8)
        waypoints.append(torch.tensor([wp_x, wp_y, wp_z], device=device))
    waypoints.append(torch.tensor([target_xy[0], target_xy[1], target_z], device=device))
    return waypoints

print(f"\n{'='*60}")
print(f"  END-TO-END MISSION (Stage 1 + PD Dock + Stage 4)")
print(f"  Stage 1: logs/gripper_wp_flight_v2/best_agent.pt")
print(f"  PD dock: analytical controller")
print(f"  Stage 4: logs/gripper_wp_loaded_v9/best_agent.pt")
print(f"{'='*60}\n")

obs, _ = env_wrapped.reset()
mission_results = []

for mission in range(NUM_MISSIONS):
    # Fresh reset
    obs, _ = env_wrapped.reset()

    start_pos = torch.tensor(STARTS[mission], device=device)
    box_pos = env.object_pos[0].clone()

    # Teleport drone to start position
    root_state = torch.zeros(1, 13, device=device)
    root_state[0, :3] = start_pos
    root_state[0, 3] = 1.0
    env.robot.write_root_state_to_sim(root_state)

    # Reset box to pedestal (use actual position from reset, not hardcoded)
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = box_pos  # position from env reset (correctly on pedestal)
    obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([0], device=device))

    # Reset env state
    env.contain_hold_count[0] = 0
    env.step_count[0] = 0
    if hasattr(env, '_stuck_count'):
        env._stuck_count[0] = 0

    # CRITICAL: clear reset_buf so step() doesn't auto-reset and overwrite teleport
    env.reset_buf[:] = False
    env.reset_terminated[:] = False
    env.reset_time_outs[:] = False

    # Dummy steps to apply teleport to physics (bypass analytical for RL control)
    env.bypass_analytical = True
    for _ in range(3):  # few steps to settle physics
        dummy = torch.zeros(1, 8, device=device)
        obs, _, _, _, _ = env_wrapped.step(dummy)
        env.reset_buf[:] = False
        env.reset_terminated[:] = False
        env.reset_time_outs[:] = False

    # Generate approach waypoints: start → WP1 → WP2 → WP3 → 50cm above box
    approach_wps = generate_waypoints(STARTS[mission], [box_pos[0].item(), box_pos[1].item()],
                                       target_z=box_pos[2].item() + 0.3)
    wp_idx = 0
    current_goal = approach_wps[wp_idx]

    # Delivery target (random, far from box)
    delivery_target = [
        (torch.rand(1).item() - 0.5) * 8.0,  # ±4m
        (torch.rand(1).item() - 0.5) * 8.0,
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

    for step in range(18000):  # 120s max
        pos_w = env.robot.data.root_pos_w[0]
        obj_pos = env.object_pos[0]
        xy_to_box = torch.norm(pos_w[:2] - obj_pos[:2]).item()

        if phase == "approach":
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

            z_above_box = pos_w[2].item() - obj_pos[2].item()
            if xy_to_box < DOCK_XY_SWITCH and z_above_box < 1.0:
                phase = "dock"
                env.bypass_analytical = False
                print(f"    [{step/150:.1f}s] APPROACH → DOCK (xy={xy_to_box:.2f}m)")

        elif phase == "dock":
            env.bypass_analytical = False
            # Hide goal marker during dock
            goal_marker.visualize(torch.tensor([[0, 0, -10]], device=device, dtype=torch.float))

            contain = env.contain_hold_count[0].item()
            gripper_angle = env.robot.data.joint_pos[0, env.plate_joint_ids[0]].item()

            if step % 150 == 0:
                print(f"      contain={contain} gripper={gripper_angle:.3f}")

            gripper_closed = gripper_angle < 0.2  # physically closed (settles ~0.18)
            if gripper_closed:
                dock_time = step / 150.0
                # Update mass for payload
                drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[0]
                env.attitude_ctrl.mass[0] = drone_mass + 0.2

                # Stage 4 takes over immediately (trained from z=0.85)
                cur_pos = pos_w.tolist()
                delivery_wps = generate_waypoints(
                    cur_pos, delivery_target, target_z=2.5, num_wps=3
                )
                wp_idx = 0
                current_goal = delivery_wps[wp_idx]
                prev_action_4d = torch.zeros(4, device=device)

                phase = "delivery"
                env.bypass_analytical = True
                print(f"    [{step/150:.1f}s] *** DOCKED *** ({dock_time:.1f}s) → DELIVERY")
                print(f"    Delivery waypoints:")
                for wi, wp in enumerate(delivery_wps):
                    tag = " (final)" if wi == len(delivery_wps)-1 else ""
                    print(f"      WP{wi+1}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}){tag}")

        elif phase == "delivery":
            env.bypass_analytical = True

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

        elif phase == "arrived":
            env.bypass_analytical = True
            # Hover at delivery point
            obs_23d = compute_loaded_obs(env, current_goal, prev_action_4d, payload_mass=0.2)
            with torch.no_grad():
                action_4d_raw = agent_s4.act(obs_23d, timestep=step, timesteps=18000)[0][0]
            prev_action_4d = action_4d_raw.clone()
            goal_marker.visualize(torch.tensor([[0, 0, -10]], device=device, dtype=torch.float))

        # Step physics
        if phase == "dock":
            dummy_8d = torch.zeros(1, 8, device=device)
            obs, reward, terminated, truncated, info = env_wrapped.step(dummy_8d)
        else:
            action_8d = torch.zeros(1, 8, device=device)
            action_8d[0, :3] = prev_action_4d[:3]
            action_8d[0, 6] = prev_action_4d[3]
            obs, reward, terminated, truncated, info = env_wrapped.step(action_8d)

        # Status
        if step % 150 == 0:
            extra = ""
            if phase == "approach":
                extra = f"wp={wp_idx+1}/{len(approach_wps)} dist_wp={dist_to_wp:.2f}"
            elif phase == "dock":
                extra = f"contain={env.contain_hold_count[0].item()}/200"
            elif phase == "delivery":
                extra = f"wp={wp_idx+1}/{len(delivery_wps)} dist_wp={dist_to_wp:.2f}"
            print(f"    [{step/150:5.1f}s] {phase:>10}  pos=({pos_w[0]:.1f},{pos_w[1]:.1f},{pos_w[2]:.1f})  "
                  f"xy_box={xy_to_box:.2f}  {extra}")

        if (terminated.view(-1) | truncated.view(-1)).any():
            if not mission_success:
                _p = env.robot.data.root_pos_w[0]
                _R = quat_to_rot_matrix(env.robot.data.root_quat_w)
                _tilt_deg = torch.acos(_R[0, 2, 2].clamp(-1, 1)).item() * 57.3
                _box_z = env.object_pos[0, 2].item()
                _xy_dist = torch.norm(_p[:2]).item()
                print(f"    [{step/150:.1f}s] TERMINATED at phase={phase}")
                print(f"      pos=({_p[0]:.1f},{_p[1]:.1f},{_p[2]:.1f}) tilt={_tilt_deg:.0f}°")
                print(f"      too_low={_p[2].item()<0.10} too_far={_xy_dist>10} too_tilted={_tilt_deg>60}")
                print(f"      box_z={_box_z:.2f} box_fell={_box_z<0.30 and env.contain_hold_count[0].item()<150}")
                print(f"      term={terminated.view(-1)[0].item()} trunc={truncated.view(-1)[0].item()}")
            break

        if phase == "arrived" and step > total_time * 150 + 450:
            break

    result = "SUCCESS" if mission_success else f"FAIL ({phase})"
    mission_results.append(mission_success)
    print(f"  Result: {result}")

n_success = sum(mission_results)
print(f"\n{'='*60}")
print(f"  MISSION SUMMARY: {n_success}/{NUM_MISSIONS}")
for i, r in enumerate(mission_results):
    print(f"    Mission {i+1}: {'SUCCESS' if r else 'FAIL'}")
print(f"{'='*60}")

env.close()
simulation_app.close()
