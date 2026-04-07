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
parser = argparse.ArgumentParser(description="Gripper-Drone PPO Training")
parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5],
                    help="Training stage (1-5)")
parser.add_argument("--substage", type=str, default=None, choices=["a", "b", "c"],
                    help="Sub-stage for Stage 3 curriculum (a/b/c)")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of parallel environments")
parser.add_argument("--max_steps", type=int, default=500_000_000,
                    help="Maximum training timesteps")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint from previous stage")
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
    print(f"  Gripper-Drone PPO Training")
    print(f"  Stage: {args.stage}" + (f" (substage {args.substage})" if args.substage else ""))
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

    # Stage 4: Dynamic box for real physics grasping
    if args.stage == 4:
        env_cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
        env_cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

    # --- Create environment ---
    env = GripperDroneEnv(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Num envs: {env.num_envs}")

    # --- Build PPO agent ---
    device = env.device
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

    # --- Train ---
    print("\nStarting training...\n")
    trainer.train()

    # --- Save final checkpoint ---
    final_path = os.path.join(log_dir, "final_agent.pt")
    agent.save(final_path)
    print(f"\nTraining complete. Final checkpoint saved to: {final_path}")

    # Cleanup
    simulation_app.close()



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
