"""Render GripperWaypointEnv flight mode with trained model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.gripper_waypoint_env import GripperWaypointEnv
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

env = GripperWaypointEnv(cfg=cfg, mode="flight")
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode="flight",
                             checkpoint_path="logs/gripper_wp_flight_v6/best_agent.pt")
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
prev_goal = env.goal_pos[0].clone()
goals = 0

for step in range(3000):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=3000)[0]
    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    if goal_marker is not None:
        goal_marker.visualize(env.goal_pos[:1])

    if torch.abs(env.goal_pos[0] - prev_goal).sum().item() > 0.01:
        goals += 1
        print(f"  [{step/150:.1f}s] GOAL #{goals}")
    prev_goal = env.goal_pos[0].clone()

    if step % 150 == 0:
        pos = env.robot.data.root_pos_w[0]
        dist = torch.norm(pos - env.goal_pos[0]).item()
        print(f"  [{step/150:5.1f}s] pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) dist={dist:.2f} goals={goals}")

    if (terminated.view(-1) | truncated.view(-1)).any():
        break

print(f"\nTotal goals: {goals}")
env.close()
simulation_app.close()
