"""Stage 1 evaluation: visualize learned policy with goal tracking."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.BASIC_FLIGHT)
cfg.scene.num_envs = 1
env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=1,
                        checkpoint_path="logs/stage1/final_agent.pt")
agent.set_running_mode("eval")

GOAL_REACH_DIST = 0.3
goals_reached = 0
max_steps = 3000

print(f"\n=== Stage 1 Evaluation ===")
print(f"  Goal threshold: {GOAL_REACH_DIST}m\n")

obs, _ = env_wrapped.reset()

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    pos = env.robot.data.root_pos_w[0]
    goal = env.goal_pos[0]
    pos_err = torch.norm(pos - goal).item()

    # Goal reached — count and generate new goal
    if pos_err < GOAL_REACH_DIST:
        goals_reached += 1
        new_xy = (torch.rand(2, device=device) * 2.0 - 1.0) * cfg.goal_pos_range_xy
        new_z = torch.rand(1, device=device) * (cfg.goal_pos_range_z[1] - cfg.goal_pos_range_z[0]) + cfg.goal_pos_range_z[0]
        env.goal_pos[0] = torch.cat([new_xy, new_z])
        print(f"  >>> GOAL {goals_reached} REACHED at step {step}! "
              f"pos_err={pos_err:.2f}m, new_goal=[{env.goal_pos[0][0]:.1f},{env.goal_pos[0][1]:.1f},{env.goal_pos[0][2]:.1f}]")

    if step % 100 == 0:
        vel = env.robot.data.root_lin_vel_w[0]
        speed = torch.norm(vel).item()
        print(f"  step {step:4d} | pos_err={pos_err:.2f}m | speed={speed:.2f}m/s | goals={goals_reached}")

    if terminated.any() or truncated.any():
        print(f"  --- Episode reset at step {step} ---")
        obs, _ = env_wrapped.reset()

print(f"\n=== Results ===")
print(f"  Total steps: {max_steps}")
print(f"  Goals reached: {goals_reached}")
print(f"  Goals per minute: {goals_reached / (max_steps / 150) * 60:.1f}")

env.close()
simulation_app.close()
