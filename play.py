"""
Gripper-Drone Evaluation / Visualization Script

Usage:
    # Evaluate Stage 1 with rendering
    python play.py --stage 1 --checkpoint logs/stage1/best_agent.pt --num_envs 16

    # Headless evaluation with metrics
    python play.py --stage 1 --checkpoint logs/stage1/best_agent.pt \
        --num_envs 256 --headless --episodes 100

    # Test robustness with increased wind
    python play.py --stage 1 --checkpoint logs/stage1/best_agent.pt \
        --wind_std 1.5 --payload 0.3
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gripper-Drone Evaluation")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5])
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
parser.add_argument("--wind_std", type=float, default=None, help="Override wind force std")
parser.add_argument("--payload", type=float, default=None, help="Override payload mass [kg]")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from collections import defaultdict

from skrl.utils import set_seed
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent


def main():
    set_seed(42)

    # --- Environment ---
    env_cfg = GripperDroneEnvCfg(stage=Stage(args.stage))
    env_cfg.scene.num_envs = args.num_envs

    # Stage 4: dynamic box for real physics grasping
    if args.stage == 4:
        env_cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
        env_cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

    # Override domain rand for robustness testing
    if args.wind_std is not None:
        env_cfg.domain_rand.wind_force_std = args.wind_std
    if args.payload is not None:
        env_cfg.domain_rand.payload_mass_range = (args.payload, args.payload)

    env = GripperDroneEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    # --- Agent ---
    device = env.device
    agent = build_ppo_agent(env=env, device=device, stage=args.stage, checkpoint_path=args.checkpoint)
    agent.set_running_mode("eval")

    # --- Evaluation loop ---
    print(f"\nEvaluating Stage {args.stage} | {args.episodes} episodes | {args.num_envs} envs\n")

    metrics = defaultdict(list)
    completed_episodes = 0
    obs, info = env.reset()

    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    episode_pos_errors = []

    while completed_episodes < args.episodes:
        with torch.no_grad():
            actions = agent.act(obs, timestep=0, timesteps=0)[0]

        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += rewards.squeeze()
        episode_lengths += 1

        # Track position errors from env extras
        if "log" in info:
            if "pos_error" in info["log"]:
                episode_pos_errors.append(info["log"]["pos_error"])

        # Collect completed episodes
        done = (terminated | truncated).squeeze()
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)

        for idx in done_ids:
            if completed_episodes >= args.episodes:
                break
            i = idx.item()
            metrics["reward"].append(episode_rewards[i].item())
            metrics["length"].append(episode_lengths[i].item())
            metrics["terminated"].append(terminated.squeeze()[i].item())
            completed_episodes += 1

            episode_rewards[i] = 0.0
            episode_lengths[i] = 0

        if completed_episodes % 10 == 0 and completed_episodes > 0:
            avg_r = np.mean(metrics["reward"][-10:])
            print(f"  Episodes: {completed_episodes}/{args.episodes}, Avg reward (last 10): {avg_r:.2f}")

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary (Stage {args.stage})")
    print(f"{'='*60}")
    print(f"  Episodes:          {len(metrics['reward'])}")
    print(f"  Mean reward:       {np.mean(metrics['reward']):.2f} +/- {np.std(metrics['reward']):.2f}")
    print(f"  Mean length:       {np.mean(metrics['length']):.0f} steps")
    print(f"  Crash rate:        {np.mean(metrics['terminated']):.1%}")
    print(f"  Survival rate:     {1 - np.mean(metrics['terminated']):.1%}")

    if episode_pos_errors:
        avg_pos_err = np.mean(episode_pos_errors)
        print(f"  Mean pos error:    {avg_pos_err:.4f} m")

    if args.wind_std is not None:
        print(f"  Wind override:     {args.wind_std} N std")
    if args.payload is not None:
        print(f"  Payload override:  {args.payload} kg")
    print(f"{'='*60}\n")

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
