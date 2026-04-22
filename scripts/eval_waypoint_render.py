"""Rendered waypoint drone evaluation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

cfg = WaypointEnvCfg(mode="flight")
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 5.0
# episode_length uses cfg default (20s) — matches training

env = WaypointDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode="flight",
                             checkpoint_path="logs/waypoint_flight_v13/best_agent.pt")
agent.set_running_mode("eval")

# Goal visualization marker
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

episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
max_steps = episode_steps * 5  # 5 episodes
obs, _ = env_wrapped.reset()
goals_reached = 0
episodes = 0
prev_goal = env.goal_pos[0].clone()

print(f"\n=== Rendered Waypoint Eval ===\n")

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    pos = env.robot.data.root_pos_w[0]
    goal = env.goal_pos[0]
    dist = torch.norm(pos - goal).item()

    # Update goal marker position
    goal_marker.visualize(env.goal_pos)

    # Goal changed?
    if torch.abs(goal - prev_goal).sum().item() > 0.01:
        goals_reached += 1
        print(f"  [{step/150:.1f}s] GOAL REACHED! #{goals_reached} → new goal "
              f"({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f})")
    prev_goal = goal.clone()

    if step % 75 == 0:
        print(f"  [{step/150:5.1f}s] pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
              f"goal=({goal[0]:.1f},{goal[1]:.1f},{goal[2]:.1f}) dist={dist:.2f}")

    if (terminated.view(-1) | truncated.view(-1)).any():
        episodes += 1
        print(f"  --- Episode {episodes} end ---\n")
        if episodes >= 5:
            break

print(f"\n{'='*40}")
print(f"  Episodes: {episodes}, Goals reached: {goals_reached}")
print(f"{'='*40}")

env.close()
simulation_app.close()
