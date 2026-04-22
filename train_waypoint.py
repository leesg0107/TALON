"""
Train waypoint-following drone (Stage 1 / Stage 4).

Separate from train.py — does NOT modify existing training code.

Usage:
  python train_waypoint.py --mode flight --num_envs 4096    # Stage 1
  python train_waypoint.py --mode loaded --num_envs 4096    # Stage 4
"""
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["flight", "loaded"], default="flight",
                    help="flight=Stage 1, loaded=Stage 4")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_steps", type=int, default=500_000_000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Must import after argparse (Isaac Lab intercepts sys.argv)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from skrl.trainers.torch import SequentialTrainer
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv
from agents.waypoint_ppo_cfg import build_waypoint_agent


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(args.seed)

    # Log directory
    mode_name = f"waypoint_{args.mode}"
    log_dir = args.log_dir or os.path.join("logs", mode_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Waypoint Drone Training")
    print(f"  Mode: {args.mode} ({'Stage 1' if args.mode == 'flight' else 'Stage 4'})")
    print(f"  Environments: {args.num_envs}")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Checkpoint: {args.checkpoint or 'None (from scratch)'}")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    # Environment config
    cfg = WaypointEnvCfg(mode=args.mode)
    cfg.scene.num_envs = args.num_envs

    if args.mode == "loaded":
        cfg.spawn_z = 1.0            # low altitude (practice climbing from near ground)
        cfg.spawn_spread = 1.0       # ±0.5m → z=0.5~1.5
        cfg.goal_range_z = (1.5, 3.0)  # goals above spawn → must climb
        print(f"  Stage 4 loaded flight:")
        print(f"    Payload: {cfg.payload_mass_range[0]:.2f}-{cfg.payload_mass_range[1]:.2f} kg")
        print(f"    Spawn z: {cfg.spawn_z}m")

    # Build env
    env = WaypointDroneEnv(cfg=cfg)
    env_wrapped = SkrlVecEnvWrapper(env)

    # Build agent
    agent = build_waypoint_agent(
        env=env_wrapped,
        device=env.device,
        mode=args.mode,
        checkpoint_path=args.checkpoint,
    )

    # Save config
    config_data = {
        "mode": args.mode,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "episode_length_s": cfg.episode_length_s,
        "obs_dim": cfg.observation_space,
        "action_dim": cfg.action_space,
        "network": "2x128 MLP (ELU)",
        "ppo_config": {
            "learning_rate": 3e-4,
            "rollouts": 24,
            "mini_batches": 24,
            "entropy_loss_scale": 0.005,
            "discount_factor": 0.99,
            "lambda": 0.95,
        },
    }
    with open(os.path.join(log_dir, "train_config.json"), 'w') as f:
        json.dump(config_data, f, indent=2)

    # SKRL SequentialTrainer counts timesteps as env.step() calls.
    # Each step advances all num_envs in parallel.
    # Effective experience = timesteps * num_envs
    timesteps = args.max_steps // args.num_envs
    print(f"  Trainer timesteps: {timesteps:,} (= {args.max_steps:,} / {args.num_envs})")

    trainer = SequentialTrainer(
        env=env_wrapped,
        agents=agent,
        cfg={
            "timesteps": timesteps,
            "headless": True,
            "disable_progressbar": False,
            "close_environment_at_exit": True,
        },
    )

    # Best checkpoint saving hook
    best_reward = [-float("inf")]
    original_post = agent.post_interaction

    def saving_post_interaction(timestep, timesteps):
        result = original_post(timestep, timesteps)
        if timestep % 1000 == 0 and timestep > 0:
            if hasattr(agent, '_cumulative_rewards') and len(agent._cumulative_rewards) > 0:
                recent = agent._cumulative_rewards[-1]
                recent = recent.item() if hasattr(recent, 'item') else recent
                if recent > best_reward[0]:
                    best_reward[0] = recent
                    agent.save(os.path.join(log_dir, "best_agent.pt"))
                    print(f"  [Best] step {timestep}: reward={recent:.1f}")
        # Periodic checkpoint every 10%
        interval = max(timesteps // 10, 1)
        if timestep % interval == 0 and timestep > 0:
            agent.save(os.path.join(log_dir, f"checkpoint_{timestep}.pt"))
            print(f"  [Checkpoint] {100*timestep/timesteps:.0f}%")
        return result

    agent.post_interaction = saving_post_interaction

    print(f"\n  Starting training...\n")
    trainer.train()

    # Save final
    agent.save(os.path.join(log_dir, "final_agent.pt"))
    print(f"\n  Final agent saved: {log_dir}/final_agent.pt")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
