"""
Gripper-Drone PPO Training Script

Unified 31D observation + 8D action architecture for ALL stages.
No dimension changes between stages — checkpoints load directly.

Usage:
    # Stage 1: Basic flight (from scratch, 31D/8D, gripper locked)
    python train.py --stage 1 --num_envs 4096 --max_steps 500_000_000

    # Stage 2: Precision approach (from Stage 1, no dim change)
    python train.py --stage 2 --num_envs 4096 --max_steps 500_000_000 \
        --checkpoint logs/stage1/final_agent.pt

    # Stage 3a: Physics-based grasping (from Stage 2, gripper unlocked)
    python train.py --stage 3 --substage a --num_envs 4096 --max_steps 300_000_000 \
        --checkpoint logs/stage2/final_agent.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# Isaac Lab imports (must be before other imports for Omniverse)
from isaaclab.app import AppLauncher

# Parse arguments before AppLauncher
parser = argparse.ArgumentParser(description="Gripper-Drone Training")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help="Training stage (1-5)")
parser.add_argument("--substage", type=str, default=None, choices=["a", "b", "c"],
                    help="Sub-stage for Stage 3 curriculum (a/b/c)")
parser.add_argument("--phase", type=int, default=None, choices=[1, 2],
                    help="Stage 3 curriculum phase (1=tight spawn dynamic, 2=full spawn dynamic)")
parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                    help="RL algorithm (ppo or sac)")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=500_000_000,
                    help="Maximum training timesteps")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint from previous stage")
parser.add_argument("--ppo_policy", type=str, default=None,
                    help="Path to PPO checkpoint for policy-only transfer to SAC")
parser.add_argument("--log_dir", type=str, default=None,
                    help="Log directory (auto-generated if not set)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--eval_interval", type=int, default=50_000_000,
                    help="Evaluate every N steps")

# AppLauncher arguments (headless, GPU, etc.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim
args.headless = True  # Training is always headless
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else (after Omniverse is initialized)
import torch
import gymnasium as gym

from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from isaaclab_rl.skrl import SkrlVecEnvWrapper

from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from agents.sac_cfg import build_sac_agent


def configure_stage3(cfg: GripperDroneEnvCfg, substage: str):
    """Configure Stage 3 sub-stage curriculum parameters."""
    if substage == "a":
        # Physics-based grasping: close-range, policy controls gripper
        cfg.auto_grasp = False           # NO auto-grasp — policy must close plates
        cfg.auto_grasp_prob = 0.0
        cfg.spawn_spread = 0.3           # close range (0.3~0.8m above object)
        cfg.approach_start_z = (0.7, 1.7)
        cfg.grasp_trigger_dist = 0.15    # gripper-to-object distance for grasp detection
        cfg.grasp_plate_threshold = 0.1  # plates must be < 0.1 rad to count as closed
        cfg.episode_length_s = 40.0      # long episode: approach ~6s + hold ~34s → 85%+ hold
    elif substage == "b":
        # Full-range physics-based grasp: start 1-3m from object
        cfg.auto_grasp = False
        cfg.auto_grasp_prob = 0.0
        cfg.spawn_spread = 2.0
        cfg.grasp_trigger_dist = 0.15
        cfg.grasp_plate_threshold = 0.1
    elif substage == "c":
        # Learned grasp: auto-grasp probability decays during training
        cfg.auto_grasp = False  # Will be toggled during training
        cfg.auto_grasp_prob = 0.0
        cfg.spawn_spread = 2.0
    else:
        raise ValueError(f"Unknown substage: {substage}")


def main():
    set_seed(args.seed)

    # --- Setup log directory ---
    stage_name = f"stage{args.stage}"
    if args.substage:
        stage_name += args.substage
    log_dir = args.log_dir or os.path.join("logs", stage_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Gripper-Drone {args.algo.upper()} Training")
    print(f"  Stage: {args.stage}" + (f" (substage {args.substage})" if args.substage else ""))
    print(f"  Algorithm: {args.algo.upper()}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Checkpoint: {args.checkpoint or 'None (from scratch)'}")
    print(f"  Log dir: {log_dir}")
    print(f"{'='*60}\n")

    # --- Build environment configuration ---
    env_cfg = GripperDroneEnvCfg(stage=Stage(args.stage))
    env_cfg.scene.num_envs = args.num_envs

    # Stage 3 sub-stage configuration
    if args.stage == 3:
        substage = args.substage or "a"
        configure_stage3(env_cfg, substage)

        # Curriculum phase configuration
        if args.phase is not None:
            env_cfg.grasping_phase = args.phase
            # Phase 1: short episode (drone spawns above box, no approach)
            # Phase 2: full episode
            env_cfg.episode_length_s = 3.0 if args.phase == 1 else 8.0
            # Both phases use dynamic box
            env_cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
            env_cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False
            print(f"  Stage 3 curriculum phase: {args.phase}")
            print(f"  Episode length: {env_cfg.episode_length_s}s")
            print(f"  Box: dynamic")

    # Stage 4: Dynamic box for real physics grasping
    if args.stage == 4:
        env_cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
        env_cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

    # --- Create environment ---
    raw_env = GripperDroneEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env)

    # SAC requires bounded action space (log_prob correction with tanh squashing)
    if args.algo == "sac":
        import gymnasium as gym
        import numpy as np
        bounded_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        # Override on the raw env before wrapping
        raw_env.action_space = bounded_action_space
        raw_env.single_action_space = bounded_action_space
        print(f"  [SAC] Action space overridden to Box(-1, 1, (8,))")

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Num envs: {env.num_envs}")

    # --- Build agent ---
    device = env.device
    if args.algo == "sac":
        agent = build_sac_agent(
            env=env,
            device=device,
            stage=args.stage,
            checkpoint_path=args.checkpoint,
            ppo_policy_path=args.ppo_policy,
        )
    else:
        agent = build_ppo_agent(
            env=env,
            device=device,
            stage=args.stage,
            substage=args.substage,
            checkpoint_path=args.checkpoint,
        )

    # --- Configure trainer ---
    # SKRL SequentialTrainer counts timesteps as individual env.step() calls.
    # Each env.step() advances all num_envs in parallel.
    # Effective experience = timesteps * num_envs
    # For 500M total experience with 4096 envs: timesteps = 500M / 4096 ≈ 122,070
    timesteps = args.max_steps // args.num_envs

    trainer_cfg = {
        "timesteps": timesteps,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": True,
    }

    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg=trainer_cfg,
    )

    # --- Stage 3c: Curriculum callback for auto-grasp probability decay ---
    if args.stage == 3 and args.substage == "c":
        _setup_stage3c_curriculum(env, agent, trainer, args.max_steps)

    # --- SAC entropy floor: prevent alpha collapse ---
    if args.algo == "sac":
        _setup_entropy_floor(agent, min_alpha=0.002)

    # --- Best checkpoint saving via post_interaction hook ---
    _setup_best_checkpoint_saving(agent, log_dir, timesteps)

    # --- Save training config for reproducibility ---
    import json
    config_record = {
        "stage": args.stage,
        "algo": args.algo,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "checkpoint": args.checkpoint,
        "ppo_policy": getattr(args, 'ppo_policy', None),
        "seed": args.seed,
        "episode_length_s": env_cfg.episode_length_s,
        "lock_gripper": env_cfg.lock_gripper,
    }
    if args.algo == "sac":
        from agents.sac_cfg import get_sac_config
        config_record["sac_config"] = get_sac_config(args.stage)
        config_record["sac_config"].pop("state_preprocessor", None)
        config_record["sac_config"].pop("state_preprocessor_kwargs", None)
        config_record["entropy_floor"] = 0.002
    else:
        from agents.ppo_cfg import get_ppo_config
        config_record["ppo_config"] = get_ppo_config(args.stage)
        config_record["ppo_config"].pop("state_preprocessor", None)
        config_record["ppo_config"].pop("state_preprocessor_kwargs", None)
        config_record["ppo_config"].pop("value_preprocessor", None)
        config_record["ppo_config"].pop("value_preprocessor_kwargs", None)
    with open(os.path.join(log_dir, "train_config.json"), "w") as f:
        json.dump(config_record, f, indent=2, default=str)
    print(f"  Config saved to {log_dir}/train_config.json")

    # --- Train ---
    print("\nStarting training...\n")
    trainer.train()

    # --- Save final checkpoint ---
    final_path = os.path.join(log_dir, "final_agent.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final checkpoint saved to: {final_path}")

    # Cleanup
    simulation_app.close()



def _setup_entropy_floor(agent, min_alpha=0.005):
    """Clamp SAC entropy coefficient to prevent collapse."""
    import math
    min_log_alpha = math.log(min_alpha)
    original_post = agent.post_interaction

    def entropy_floor_post(timestep, timesteps):
        with torch.no_grad():
            if agent.log_entropy_coefficient.item() < min_log_alpha:
                agent.log_entropy_coefficient.fill_(min_log_alpha)
        return original_post(timestep, timesteps)

    agent.post_interaction = entropy_floor_post
    print(f"  [SAC] Entropy floor set: min_alpha={min_alpha}")


def _setup_best_checkpoint_saving(agent, log_dir, total_timesteps):
    """Save best checkpoint based on total reward, and periodic checkpoints."""
    best_reward = [-float("inf")]
    original_post = agent.post_interaction

    def saving_post_interaction(timestep, timesteps):
        result = original_post(timestep, timesteps)

        # Check every 1000 timesteps
        if timestep % 1000 == 0 and timestep > 0:
            # Get recent reward from agent's tracking
            if hasattr(agent, '_cumulative_rewards') and len(agent._cumulative_rewards) > 0:
                recent = agent._cumulative_rewards[-1].item() if hasattr(agent._cumulative_rewards[-1], 'item') else agent._cumulative_rewards[-1]
            else:
                recent = None

            if recent is not None and recent > best_reward[0]:
                best_reward[0] = recent
                best_path = os.path.join(log_dir, "best_agent.pt")
                agent.save(best_path)
                print(f"  [Best] step {timestep}: reward={recent:.1f} → saved {best_path}")

        # Periodic checkpoint every 10% of training
        interval = max(total_timesteps // 10, 1)
        if timestep % interval == 0 and timestep > 0:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{timestep}.pt")
            agent.save(ckpt_path)
            pct = 100 * timestep / total_timesteps
            print(f"  [Checkpoint] step {timestep} ({pct:.0f}%) → saved {ckpt_path}")

        return result

    agent.post_interaction = saving_post_interaction


def _setup_stage3c_curriculum(env, agent, trainer, max_steps):
    """Setup gradual auto-grasp probability decay for Stage 3c.

    Probability schedule:
      Phase 0: auto_grasp=True,  prob=1.0 (always auto-grasp)
      Phase 1: auto_grasp=True,  prob=0.5 (50% auto, 50% learned)
      Phase 2: auto_grasp=False, prob=0.0 (fully learned grasp)

    Implemented by wrapping agent.post_interaction (called by SKRL trainer).
    """
    total_phases = 3
    steps_per_phase = max_steps // total_phases
    probs = [1.0, 0.5, 0.0]

    original_post_interaction = agent.post_interaction

    def curriculum_post_interaction(timestep, timesteps):
        phase = min(int(timestep / steps_per_phase), total_phases - 1)

        unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        unwrapped.cfg.auto_grasp = (phase < 2)
        unwrapped.cfg.auto_grasp_prob = probs[phase]

        if steps_per_phase > 0 and timestep % max(steps_per_phase // 10, 1) == 0:
            print(f"  [Curriculum] Step {timestep:,}: phase={phase}, "
                  f"auto_grasp={phase < 2}, prob={probs[phase]}")

        return original_post_interaction(timestep, timesteps)

    agent.post_interaction = curriculum_post_interaction


if __name__ == "__main__":
    main()
