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
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
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

    # Stage 3 curriculum phase (dynamic box, tight spawn in phase 1)
    if args.stage == 3 and args.phase is not None:
        env_cfg.grasping_phase = args.phase
        env_cfg.episode_length_s = 3.0 if args.phase == 1 else 8.0
        env_cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
        env_cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

    # Override domain rand for robustness testing
    if args.wind_std is not None:
        env_cfg.domain_rand.wind_force_std = args.wind_std
    if args.payload is not None:
        env_cfg.domain_rand.payload_mass_range = (args.payload, args.payload)

    raw_env = GripperDroneEnv(cfg=env_cfg)

    # SAC requires bounded action space — set BEFORE wrapping
    if args.algo == "sac":
        import gymnasium as gym
        raw_env.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        raw_env.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    env = SkrlVecEnvWrapper(raw_env)

    # --- Agent ---
    device = env.device
    if args.algo == "sac":
        from agents.sac_cfg import build_sac_agent
        agent = build_sac_agent(env=env, device=device, stage=args.stage, checkpoint_path=args.checkpoint)
    else:
        agent = build_ppo_agent(env=env, device=device, stage=args.stage, checkpoint_path=args.checkpoint)
    agent.set_running_mode("eval")

    # --- Evaluation loop ---
    print(f"\nEvaluating Stage {args.stage} | {args.episodes} episodes | {args.num_envs} envs\n")

    metrics = defaultdict(list)
    completed_episodes = 0
    obs, info = env.reset()

    episode_rewards = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    min_pos_error = torch.full((args.num_envs,), float('inf'), device=device)
    episode_pos_errors = []

    while completed_episodes < args.episodes:
        with torch.no_grad():
            actions = agent.act(obs, timestep=0, timesteps=0)[0]

        obs, rewards, terminated, truncated, info = env.step(actions)
        episode_rewards += rewards.squeeze()
        episode_lengths += 1

        # Track per-env position error
        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        if hasattr(unwrapped, 'robot') and hasattr(unwrapped, 'goal_pos'):
            from envs.drone_env import quat_to_rot_matrix
            pos_w = unwrapped.robot.data.root_pos_w
            quat_w = unwrapped.robot.data.root_quat_w
            R = quat_to_rot_matrix(quat_w)
            gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=device)
            gripper_pos = pos_w + torch.bmm(R, gripper_offset.expand(args.num_envs, -1).unsqueeze(-1)).squeeze(-1)
            pos_err = torch.norm(gripper_pos - unwrapped.goal_pos, dim=-1)
            min_pos_error = torch.min(min_pos_error, pos_err)

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
            metrics["min_pos_error"].append(min_pos_error[i].item())
            metrics["reached_0.5m"].append(1.0 if min_pos_error[i].item() < 0.5 else 0.0)
            metrics["reached_0.3m"].append(1.0 if min_pos_error[i].item() < 0.3 else 0.0)
            metrics["reached_0.1m"].append(1.0 if min_pos_error[i].item() < 0.1 else 0.0)
            completed_episodes += 1

            episode_rewards[i] = 0.0
            episode_lengths[i] = 0
            min_pos_error[i] = float('inf')

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

    if metrics.get("min_pos_error"):
        print(f"  Min pos error:     {np.mean(metrics['min_pos_error']):.4f} m (avg of per-episode best)")
        print(f"  Reached <0.5m:     {np.mean(metrics['reached_0.5m']):.1%}")
        print(f"  Reached <0.3m:     {np.mean(metrics['reached_0.3m']):.1%}")
        print(f"  Reached <0.1m:     {np.mean(metrics['reached_0.1m']):.1%}")

    if args.wind_std is not None:
        print(f"  Wind override:     {args.wind_std} N std")
    if args.payload is not None:
        print(f"  Payload override:  {args.payload} kg")
    print(f"{'='*60}\n")

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
