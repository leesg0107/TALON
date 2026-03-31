"""Stage 2 evaluation: precision approach — descend to object from above."""
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

cfg = GripperDroneEnvCfg(stage=Stage.PRECISION_APPROACH)
cfg.scene.num_envs = 1
env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=2,
                        checkpoint_path="runs/26-03-23_23-06-40-055753_PPO/checkpoints/best_agent.pt")
agent.set_running_mode("eval")

max_steps = 3000
reach_count = 0
REACH_DIST = 0.15  # 15cm from goal (gripper height)

print(f"\n=== Stage 2 Evaluation: Precision Approach ===")
print(f"  Goal: descend to gripper height above object")
print(f"  Reach threshold: {REACH_DIST}m\n")

obs, _ = env_wrapped.reset()

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    pos = env.robot.data.root_pos_w[0].cpu()
    goal = env.goal_pos[0].cpu()
    xy_err = torch.norm(pos[:2] - goal[:2]).item()
    alt_err = abs(pos[2].item() - goal[2].item())
    total_err = torch.norm(pos - goal).item()

    if step % 75 == 0:
        vel = env.robot.data.root_lin_vel_w[0].cpu()
        print(f"  step {step:4d} | xy_err={xy_err:.2f}m | alt_err={alt_err:.2f}m | "
              f"total={total_err:.2f}m | vz={vel[2]:.2f}m/s | "
              f"pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}] "
              f"goal=[{goal[0]:.1f},{goal[1]:.1f},{goal[2]:.1f}]")

    if total_err < REACH_DIST:
        reach_count += 1
        if reach_count == 1:
            print(f"  >>> FIRST REACH at step {step}! err={total_err:.3f}m")

    if terminated.any() or truncated.any():
        print(f"  --- Episode reset at step {step} ---")
        obs, _ = env_wrapped.reset()

print(f"\n=== Results ===")
print(f"  Steps within {REACH_DIST}m: {reach_count}/{max_steps} ({100*reach_count/max_steps:.1f}%)")

env.close()
simulation_app.close()
