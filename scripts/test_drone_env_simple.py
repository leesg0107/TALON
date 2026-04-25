"""Minimal test: can the flight model reach goals in GripperDroneEnv?

NO phase transitions, NO complex pipeline. Just:
1. GripperDroneEnv with bypass_analytical=True
2. compute_waypoint_obs() for obs
3. agent outputs 4D action
4. Map to 8D and step

If this works → eval_mission.py pipeline is the problem.
If this fails → compute_waypoint_obs() or env mapping is the problem.
"""
import sys, os, math
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
from envs.waypoint_cfg import WaypointEnvCfg
import gymnasium as gym

# --- Env: GripperDroneEnv (same as eval_mission.py) ---
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 1
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

# --- Agent ---
class FakeEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(-float('inf'), float('inf'), (22,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,))
        self.num_envs = 1

agent = build_waypoint_agent(env=FakeEnv(), device=device, mode="flight",
                             checkpoint_path="logs/gripper_wp_flight_v6/best_agent.pt")
agent.set_running_mode("eval")

# --- Goal marker ---
goal_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/GoalMarker",
    markers={"goal": sim_utils.SphereCfg(radius=0.15,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.6))},
)
goal_marker = VisualizationMarkers(goal_marker_cfg)

# --- Obs computation: SAME as eval_mission.py ---
_dr = WaypointEnvCfg().domain_rand

def compute_obs(env, goal_pos, prev_action_4d):
    vel_b = env.robot.data.root_lin_vel_b[0]
    ang_vel_b = env.robot.data.root_ang_vel_b[0]
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    pos_w = env.robot.data.root_pos_w[0]
    goal_err_w = goal_pos - pos_w
    goal_b = R[0].T @ goal_err_w
    R_flat = R[0].reshape(9)
    prev_norm = prev_action_4d.clone()
    prev_norm[:3] /= 8.0
    prev_norm[3] /= math.pi
    vel_b_n = vel_b + _dr.vel_noise_std * torch.randn_like(vel_b)
    goal_b_n = goal_b + _dr.pos_noise_std * torch.randn_like(goal_b)
    return torch.cat([vel_b_n, ang_vel_b, R_flat, goal_b_n, prev_norm]).unsqueeze(0)

# --- Reset and set bypass ---
obs, _ = env_wrapped.reset()
env.bypass_analytical = True

# Clear any reset issues
for _ in range(3):
    env_wrapped.step(torch.zeros(1, 8, device=device))
    env.reset_buf[:] = False
    env.reset_terminated[:] = False
    env.reset_time_outs[:] = False

# --- Generate random goals (SAME pattern as training trajectory) ---
prev_action_4d = torch.zeros(4, device=device)
goals_reached = 0
goal_pos = env.scene.env_origins[0] + torch.tensor([1.5, 1.0, 2.5], device=device)

print(f"\n=== Simple Goal Test in GripperDroneEnv ===")
print(f"  bypass_analytical = True")
print(f"  Using compute_waypoint_obs() (same as eval_mission.py)")
print(f"  Agent: gripper_wp_flight_v6/best_agent.pt\n")

for step in range(4500):  # 30s
    pos_w = env.robot.data.root_pos_w[0]
    dist = torch.norm(pos_w - goal_pos).item()

    # Compute obs and act
    obs_22d = compute_obs(env, goal_pos, prev_action_4d)
    with torch.no_grad():
        action_4d = agent.act(obs_22d, timestep=step, timesteps=4500)[0][0]
    prev_action_4d = action_4d.clone()

    # Map 4D → 8D (same as eval_mission.py)
    action_8d = torch.zeros(1, 8, device=device)
    action_8d[0, :3] = action_4d[:3]
    action_8d[0, 6] = action_4d[3]

    # Visualize
    goal_marker.visualize(goal_pos.unsqueeze(0))

    # Step
    obs, reward, terminated, truncated, info = env_wrapped.step(action_8d)
    env.reset_buf[:] = False
    env.reset_terminated[:] = False
    env.reset_time_outs[:] = False

    # Goal reached → new random goal
    if dist < 0.3:
        goals_reached += 1
        # New random goal (training-like pattern)
        angle = torch.rand(1).item() * 2 * math.pi
        d = 0.8 + torch.rand(1).item() * 1.5
        origin = env.scene.env_origins[0]
        goal_pos = pos_w.clone()
        goal_pos[0] += d * math.cos(angle)
        goal_pos[1] += d * math.sin(angle)
        goal_pos[2] = origin[2].item() + 1.0 + torch.rand(1).item() * 2.5
        prev_action_4d = torch.zeros(4, device=device)
        print(f"  [{step/150:.1f}s] GOAL #{goals_reached} reached! dist={dist:.2f}")

    if step % 150 == 0:
        print(f"  [{step/150:5.1f}s] pos=({pos_w[0]:.1f},{pos_w[1]:.1f},{pos_w[2]:.1f}) "
              f"dist={dist:.2f} goals={goals_reached}")

print(f"\n  Total goals in 30s: {goals_reached}")
print(f"  {'PASS' if goals_reached >= 5 else 'FAIL'}: model {'works' if goals_reached >= 5 else 'BROKEN'} in GripperDroneEnv")

env.close()
simulation_app.close()
