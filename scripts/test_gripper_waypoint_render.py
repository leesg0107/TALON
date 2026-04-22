"""Render GripperWaypointEnv loaded mode — debug PD dock."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.gripper_waypoint_env import GripperWaypointEnv
from envs.drone_env import quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 20.0
cfg.lock_gripper = True
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperWaypointEnv(cfg=cfg, mode="loaded")
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode="loaded",
                             checkpoint_path="logs/gripper_wp_loaded_v7/best_agent.pt")
agent.set_running_mode("eval")

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

obs, _ = env_wrapped.reset()

print(f"\n=== GripperWaypointEnv Loaded v7 ===")
print(f"  dock_phase: {env._dock_phase[0].item()}")
print(f"  drone pos: {env.robot.data.root_pos_w[0].tolist()}")
print(f"  box pos: {env.grasp_object.data.root_pos_w[0].tolist()}")
# Check pedestal
try:
    pedestal = env.scene["pedestal"]
    print(f"  pedestal pos: {pedestal.data.root_pos_w[0].tolist()}")
    print(f"  pedestal exists: True")
except:
    print(f"  pedestal exists: False ← THIS IS THE PROBLEM")
print()

prev_goal = None
goals_reached = 0
total_reward = 0.0

for step in range(3000):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=3000)[0]
    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    if goal_marker is not None:
        goal_marker.visualize(env.goal_pos[:1])

    pos = env.robot.data.root_pos_w[0]
    box_pos = env.grasp_object.data.root_pos_w[0]
    R = quat_to_rot_matrix(env.robot.data.root_quat_w)
    tilt_deg = torch.acos(R[0, 2, 2].clamp(-1, 1)).item() * 57.3
    goal = env.goal_pos[0]
    goal_dist = torch.norm(pos - goal).item()
    total_reward += reward[0].item()

    # Track goal changes
    if prev_goal is not None and torch.abs(goal - prev_goal).sum().item() > 0.01:
        goals_reached += 1
        print(f"  >>> GOAL #{goals_reached} reached at step {step} ({step/150:.1f}s)")
    prev_goal = goal.clone()

    if step % 75 == 0:
        print(f"  step {step:>4} | dock={env._dock_phase[0].item()} | "
              f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | "
              f"box=({box_pos[0]:.2f},{box_pos[1]:.2f},{box_pos[2]:.2f}) | "
              f"tilt={tilt_deg:.0f}° | "
              f"contain={env.contain_hold_count[0].item()} | "
              f"reward={reward[0].item():.2f} | "
              f"goal_dist={goal_dist:.2f} | "
              f"goals={goals_reached}")

    if (terminated.view(-1) | truncated.view(-1)).any():
        reason = "TRUNCATED (timeout)" if truncated.view(-1).any() else "TERMINATED"
        print(f"\n  {reason} at step {step} ({step/150:.1f}s)")
        print(f"    pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
        print(f"    tilt={tilt_deg:.0f}°")
        print(f"    box_z={box_pos[2]:.2f}")
        print(f"    contain={env.contain_hold_count[0].item()}")
        print(f"    goals_reached={goals_reached}")
        print(f"    total_reward={total_reward:.1f}")
        print(f"    too_low={pos[2].item()<0.15}")
        print(f"    too_far={torch.norm(pos[:2]).item()>10}")
        print(f"    too_tilted={tilt_deg>60}")
        print(f"    box_fell={box_pos[2].item()<0.30 and env.contain_hold_count[0].item()<150}")
        break

env.close()
simulation_app.close()
