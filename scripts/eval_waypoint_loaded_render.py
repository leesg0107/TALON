"""Rendered eval for Stage 4 loaded flight (physical box in gripper)."""
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

cfg = WaypointEnvCfg(mode="loaded")
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 5.0

env = WaypointDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode="loaded",
                             checkpoint_path="logs/waypoint_loaded_v7/best_agent.pt")
agent.set_running_mode("eval")

# Goal marker
goal_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/GoalMarker",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), opacity=0.6,
            ),
        ),
    },
)
goal_marker = VisualizationMarkers(goal_marker_cfg)

episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
obs, _ = env_wrapped.reset()

print(f"\n=== Stage 4 Loaded Flight — Delivery Route ===\n")

# Delivery route: fixed waypoints in a direction (simulating delivery)
delivery_route = [
    torch.tensor([1.0, 0.5, 2.5], device=device),
    torch.tensor([2.0, 1.0, 2.5], device=device),
    torch.tensor([3.0, 1.5, 2.5], device=device),
    torch.tensor([4.0, 2.0, 2.0], device=device),
]

wp_idx = 0
env.goal_pos[0] = delivery_route[wp_idx]
prev_goal = env.goal_pos[0].clone()

print(f"  Route:")
for i, wp in enumerate(delivery_route):
    print(f"    WP{i+1}: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")
print()

max_steps = 4500  # 30 seconds

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    pos = env.robot.data.root_pos_w[0]
    goal = env.goal_pos[0]
    dist = torch.norm(pos - goal).item()

    goal_marker.visualize(goal.unsqueeze(0))

    # Check waypoint reached (env regens at 0.3m, override with our route)
    if dist < 0.3 and wp_idx < len(delivery_route) - 1:
        wp_idx += 1
        env.goal_pos[0] = delivery_route[wp_idx]
        print(f"  [{step/150:.1f}s] WP{wp_idx} reached → WP{wp_idx+1} ({delivery_route[wp_idx][0]:.1f},{delivery_route[wp_idx][1]:.1f},{delivery_route[wp_idx][2]:.1f})")
    elif dist < 0.3 and wp_idx == len(delivery_route) - 1:
        print(f"  [{step/150:.1f}s] *** DELIVERY COMPLETE ***")
        # Hold for 2 seconds then end
        if step % 150 == 0:
            pass
        if step > 300:  # at least after some time
            break

    if step % 75 == 0:
        box_pos = env.grasp_box.data.root_pos_w[0] if env.grasp_box else pos
        box_dist = torch.norm(pos - box_pos).item()
        R = __import__('controllers.drone_ctrl', fromlist=['quat_to_rot_matrix']).quat_to_rot_matrix(env.robot.data.root_quat_w)
        tilt_deg = __import__('torch').acos(R[0, 2, 2].clamp(-1, 1)).item() * 57.3
        print(f"  [{step/150:5.1f}s] pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
              f"dist={dist:.2f} tilt={tilt_deg:.0f}° box_off={box_dist:.3f}")

    if (terminated.view(-1) | truncated.view(-1)).any():
        tilt_deg = __import__('torch').acos(
            __import__('controllers.drone_ctrl', fromlist=['quat_to_rot_matrix']).quat_to_rot_matrix(
                env.robot.data.root_quat_w)[0, 2, 2].clamp(-1, 1)).item() * 57.3
        xy_dist = torch.norm(pos[:2]).item()
        print(f"  [{step/150:.1f}s] TERMINATED tilt={tilt_deg:.0f}° xy_dist={xy_dist:.1f}m alt={pos[2]:.1f}m")
        break

env.close()
simulation_app.close()
