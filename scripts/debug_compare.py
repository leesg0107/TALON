"""Debug: compare obs/action between GripperWaypointEnv and eval_mission's compute_waypoint_obs.

Runs GripperWaypointEnv for steps, then replicates in eval_mission style to find differences.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.gripper_waypoint_env import GripperWaypointEnv
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import gymnasium as gym

# ============================================================
# Setup: GripperWaypointEnv (training env, 1 env)
# ============================================================
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 20.0
cfg.lock_gripper = True
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperWaypointEnv(cfg=cfg, mode="flight")
env_w = SkrlVecEnvWrapper(env)
device = env.device

# Agent loaded via proper env (standalone path)
agent_proper = build_waypoint_agent(env=env_w, device=device, mode="flight",
                                    checkpoint_path="logs/gripper_wp_flight_v6/final_agent.pt")
agent_proper.set_running_mode("eval")

# Agent loaded via FakeEnv (eval_mission path)
obs_space = gym.spaces.Box(-float('inf'), float('inf'), (22,))
act_space = gym.spaces.Box(-1.0, 1.0, (4,))

class FakeEnv:
    def __init__(self):
        self.observation_space = obs_space
        self.action_space = act_space
        self.num_envs = 1

agent_fake = build_waypoint_agent(env=FakeEnv(), device=device, mode="flight",
                                  checkpoint_path="logs/gripper_wp_flight_v6/final_agent.pt")
agent_fake.set_running_mode("eval")

# ============================================================
# Compare scalers
# ============================================================
s1 = agent_proper._state_preprocessor
s2 = agent_fake._state_preprocessor
print("=== Scaler comparison ===")
print(f"Scaler1 (proper) mean[:5]: {s1.running_mean[:5].tolist()}")
print(f"Scaler2 (fake)   mean[:5]: {s2.running_mean[:5].tolist()}")
mean_diff = torch.abs(s1.running_mean.cpu() - s2.running_mean.cpu())
var_diff = torch.abs(s1.running_variance.cpu() - s2.running_variance.cpu())
print(f"Mean max diff: {mean_diff.max().item():.10f}")
print(f"Var max diff:  {var_diff.max().item():.10f}")
print(f"Count1: {s1.current_count.item():.0f}")
print(f"Count2: {s2.current_count.item():.0f}")

# ============================================================
# Compare policy weights
# ============================================================
p1 = agent_proper.policy.state_dict()
p2 = agent_fake.policy.state_dict()
print("\n=== Policy weight comparison ===")
for k in p1:
    diff = torch.abs(p1[k].cpu().float() - p2[k].cpu().float()).max().item()
    print(f"  {k}: max diff = {diff:.10f}")

# ============================================================
# Run 10 steps, compare obs and actions from both agents
# ============================================================
print("\n=== Step comparison (same obs → both agents) ===\n")

obs, _ = env_w.reset()
prev_action = torch.zeros(4, device=device)

for step in range(10):
    # Get env's obs (from training pipeline)
    raw_obs = obs["policy"] if isinstance(obs, dict) else obs

    # Also compute manual obs (eval_mission style)
    goal = env.goal_pos[0]
    vel_b = env.robot.data.root_lin_vel_b[0]
    ang_vel_b = env.robot.data.root_ang_vel_b[0]
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    pos_w = env.robot.data.root_pos_w[0]
    goal_err_w = goal - pos_w
    goal_b = R[0].T @ goal_err_w
    R_flat = R[0].reshape(9)
    prev_norm = prev_action.clone()
    prev_norm[:3] /= 8.0
    prev_norm[3] /= math.pi
    manual_obs = torch.cat([vel_b, ang_vel_b, R_flat, goal_b, prev_norm]).unsqueeze(0)

    # Compare raw obs (before scaler) — IGNORE noise dimensions
    diff = torch.abs(raw_obs[0] - manual_obs[0])
    # vel_b has noise in training, so skip [0:3]
    # goal_b has noise in training, so skip [15:18]
    mask = torch.ones(22, dtype=torch.bool)
    mask[:3] = False   # vel_b noisy
    mask[15:18] = False  # goal_b noisy
    clean_diff = diff[mask]

    print(f"Step {step}:")
    print(f"  pos=({pos_w[0]:.2f},{pos_w[1]:.2f},{pos_w[2]:.2f})  goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f})")
    print(f"  Raw obs diff (excl noise dims): max={clean_diff.max():.6f} mean={clean_diff.mean():.6f}")

    # Get actions from both agents on SAME obs
    with torch.no_grad():
        act_proper = agent_proper.act(obs, timestep=step, timesteps=100)[0]
        act_fake = agent_fake.act(obs, timestep=step, timesteps=100)[0]

    act_diff = torch.abs(act_proper[0] - act_fake[0])
    print(f"  Action proper: [{act_proper[0,0]:.4f}, {act_proper[0,1]:.4f}, {act_proper[0,2]:.4f}, {act_proper[0,3]:.4f}]")
    print(f"  Action fake:   [{act_fake[0,0]:.4f}, {act_fake[0,1]:.4f}, {act_fake[0,2]:.4f}, {act_fake[0,3]:.4f}]")
    print(f"  Action diff:   max={act_diff.max():.6f}")

    # Step env
    obs, _, _, _, _ = env_w.step(act_proper)
    prev_action = act_proper[0].clone()

    # Also check: what does manual obs look like after scaler?
    scaled_proper = s1(raw_obs)
    scaled_fake = s2(raw_obs)  # same raw obs through different scaler
    scale_diff = torch.abs(scaled_proper[0] - scaled_fake[0])
    print(f"  Scaled obs diff: max={scale_diff.max():.6f}")
    print()

env.close()
simulation_app.close()
