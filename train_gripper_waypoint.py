"""
Train waypoint flight in GripperDroneEnv (eliminates transfer gap).

Usage:
  python train_gripper_waypoint.py --mode flight --num_envs 4096    # Stage 1
  python train_gripper_waypoint.py --mode loaded --num_envs 4096    # Stage 4
"""
import argparse
import os
import json

# Ensure working directory is project root (for relative paths)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["flight", "loaded"], default="flight")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_steps", type=int, default=1_000_000_000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--warm_start", type=str, default=None, help="Flight checkpoint for warm-start (22D→23D)")
parser.add_argument("--reset_std", action="store_true", help="Reset log_std for re-exploration")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from skrl.trainers.torch import SequentialTrainer
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.gripper_waypoint_env import GripperWaypointEnv
from agents.waypoint_ppo_cfg import build_waypoint_agent


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(args.seed)

    mode_name = f"gripper_waypoint_{args.mode}"
    log_dir = args.log_dir or os.path.join("models", mode_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Gripper Waypoint Training (in GripperDroneEnv)")
    print(f"  Mode: {args.mode}")
    print(f"  Envs: {args.num_envs}")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 5.0
    cfg.episode_length_s = 20.0
    cfg.lock_gripper = True
    cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
    cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

    env = GripperWaypointEnv(cfg=cfg, mode=args.mode)
    env_wrapped = SkrlVecEnvWrapper(env)

    agent = build_waypoint_agent(
        env=env_wrapped, device=env.device, mode=args.mode,
        checkpoint_path=args.checkpoint,
        warm_start_flight=args.warm_start,
    )

    # Reset log_std if loading checkpoint (allow re-exploration for new reward)
    if (args.checkpoint or args.warm_start) and args.reset_std:
        with torch.no_grad():
            agent.models["policy"].log_std.fill_(0.0)  # std=1.0
        print(f"  Reset log_std to 0.0 (std=1.0) for re-exploration")

    timesteps = args.max_steps // args.num_envs
    print(f"  Trainer timesteps: {timesteps:,}")

    trainer = SequentialTrainer(
        env=env_wrapped, agents=agent,
        cfg={"timesteps": timesteps, "headless": True,
             "disable_progressbar": False, "close_environment_at_exit": True},
    )

    # Best checkpoint hook
    best_reward = [-float("inf")]
    original_post = agent.post_interaction

    def saving_post(timestep, timesteps):
        result = original_post(timestep, timesteps)
        if timestep % 1000 == 0 and timestep > 0:
            if hasattr(agent, '_cumulative_rewards') and len(agent._cumulative_rewards) > 0:
                recent = agent._cumulative_rewards[-1]
                recent = recent.item() if hasattr(recent, 'item') else recent
                if recent > best_reward[0]:
                    best_reward[0] = recent
                    agent.save(os.path.join(log_dir, "best_agent.pt"))
                    print(f"  [Best] step {timestep}: reward={recent:.1f}")
        interval = max(timesteps // 10, 1)
        if timestep % interval == 0 and timestep > 0:
            agent.save(os.path.join(log_dir, f"checkpoint_{timestep}.pt"))
            print(f"  [Checkpoint] {100*timestep/timesteps:.0f}%")
        return result

    agent.post_interaction = saving_post

    with open(os.path.join(log_dir, "train_config.json"), 'w') as f:
        json.dump({"mode": args.mode, "num_envs": args.num_envs,
                   "max_steps": args.max_steps, "env": "GripperWaypointEnv"}, f, indent=2)

    print(f"\n  Starting training...\n")
    trainer.train()

    agent.save(os.path.join(log_dir, "final_agent.pt"))
    print(f"\n  Final saved: {log_dir}/final_agent.pt")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
