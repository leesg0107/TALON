"""Simple SAC evaluation/rendering script."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import numpy as np
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.sac_cfg import build_sac_agent

env_cfg = GripperDroneEnvCfg(stage=Stage(args.stage))
env_cfg.scene.num_envs = args.num_envs
env_cfg.scene.env_spacing = 10.0

raw_env = GripperDroneEnv(cfg=env_cfg)
raw_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
raw_env.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
env = SkrlVecEnvWrapper(raw_env)

device = env.device
agent = build_sac_agent(env=env, device=device, stage=args.stage, checkpoint_path=args.checkpoint)
agent.set_running_mode("eval")

print(f"\n  SAC Eval: Stage {args.stage}, {args.num_envs} envs")
print(f"  Checkpoint: {args.checkpoint}\n")

obs, _ = env.reset()
for step in range(9000):  # 60s at 150Hz
    with torch.no_grad():
        # timestep=random_timesteps to skip random action phase
        action = agent.act(obs, timestep=99999, timesteps=9000)[0]
    obs, reward, terminated, truncated, info = env.step(action)

simulation_app.close()
