"""
Curriculum Manager for Gripper-Drone multi-stage training.

Orchestrates the full 5-stage curriculum pipeline:
  Stage 1:  Basic flight           (from scratch)
  Stage 2:  Precision approach     (from Stage 1)
  Stage 3a: Auto-grasp close       (from Stage 2)
  Stage 3b: Auto-grasp full-range  (from Stage 3a)
  Stage 3c: Learned grasp          (from Stage 3b)
  Stage 4:  Loaded flight          (from Stage 1)
  Stage 5:  Release                (from Stage 3c)

Usage:
    # Run the full curriculum automatically
    python curriculum.py --num_envs 4096

    # Resume from a specific stage
    python curriculum.py --num_envs 4096 --start_stage 3 --start_substage b \
        --checkpoint logs/stage3a/best_agent.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    stage: int
    substage: str | None
    max_steps: int
    checkpoint_from: str | None  # stage name to load checkpoint from
    success_metric: str
    success_threshold: float


# Full curriculum pipeline definition
CURRICULUM = [
    StageConfig(
        stage=1, substage=None, max_steps=500_000_000,
        checkpoint_from=None,
        success_metric="pos_error", success_threshold=0.05,
    ),
    StageConfig(
        stage=2, substage=None, max_steps=500_000_000,
        checkpoint_from="stage1",
        success_metric="xy_err", success_threshold=0.03,
    ),
    StageConfig(
        stage=3, substage="a", max_steps=300_000_000,
        checkpoint_from="stage2",
        success_metric="grasp_rate", success_threshold=0.90,
    ),
    StageConfig(
        stage=3, substage="b", max_steps=500_000_000,
        checkpoint_from="stage3a",
        success_metric="grasp_rate", success_threshold=0.85,
    ),
    StageConfig(
        stage=3, substage="c", max_steps=800_000_000,
        checkpoint_from="stage3b",
        success_metric="grasp_rate", success_threshold=0.80,
    ),
    StageConfig(
        stage=4, substage=None, max_steps=500_000_000,
        checkpoint_from="stage1",
        success_metric="pos_error", success_threshold=0.10,
    ),
    StageConfig(
        stage=5, substage=None, max_steps=500_000_000,
        checkpoint_from="stage3c",
        success_metric="release_rate", success_threshold=0.90,
    ),
]


def stage_name(cfg: StageConfig) -> str:
    name = f"stage{cfg.stage}"
    if cfg.substage:
        name += cfg.substage
    return name


def find_checkpoint(log_base: str, stage_key: str) -> str | None:
    """Find the best checkpoint for a given stage."""
    stage_dir = os.path.join(log_base, stage_key)
    # Prefer best_agent.pt, fall back to final_agent.pt
    for fname in ["best_agent.pt", "final_agent.pt"]:
        path = os.path.join(stage_dir, fname)
        if os.path.exists(path):
            return path
    return None


def run_stage(cfg: StageConfig, num_envs: int, log_base: str) -> bool:
    """Run a single training stage. Returns True if successful."""
    name = stage_name(cfg)
    log_dir = os.path.join(log_base, name)

    # Find checkpoint from previous stage
    checkpoint_arg = []
    if cfg.checkpoint_from:
        ckpt = find_checkpoint(log_base, cfg.checkpoint_from)
        if ckpt is None:
            print(f"[ERROR] Cannot find checkpoint for {cfg.checkpoint_from}!")
            print(f"        Expected in: {os.path.join(log_base, cfg.checkpoint_from)}")
            return False
        checkpoint_arg = ["--checkpoint", ckpt]

    # Build command
    cmd = [
        sys.executable, "train.py",
        "--stage", str(cfg.stage),
        "--num_envs", str(num_envs),
        "--max_steps", str(cfg.max_steps),
        "--log_dir", log_dir,
        "--headless",
    ]
    if cfg.substage:
        cmd.extend(["--substage", cfg.substage])
    cmd.extend(checkpoint_arg)

    print(f"\n{'='*60}")
    print(f"  CURRICULUM: Starting {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Max steps: {cfg.max_steps:,}")
    print(f"  Success: {cfg.success_metric} < {cfg.success_threshold}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Run training
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ERROR] {name} training failed with code {result.returncode}")
        return False

    # Verify checkpoint was created
    if find_checkpoint(log_base, name) is None:
        print(f"\n[ERROR] No checkpoint found after {name} training!")
        return False

    print(f"\n[OK] {name} training completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gripper-Drone Curriculum Manager")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--log_base", type=str, default="logs")
    parser.add_argument("--start_stage", type=int, default=1,
                        help="Stage to start from (for resuming)")
    parser.add_argument("--start_substage", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint for starting stage")
    args = parser.parse_args()

    os.makedirs(args.log_base, exist_ok=True)

    # Find starting point in curriculum
    start_name = f"stage{args.start_stage}"
    if args.start_substage:
        start_name += args.start_substage

    started = False
    for cfg in CURRICULUM:
        name = stage_name(cfg)

        if not started:
            if name == start_name:
                started = True
                # Override checkpoint if provided
                if args.checkpoint:
                    cfg.checkpoint_from = None  # Will use args.checkpoint directly
            else:
                continue

        success = run_stage(cfg, args.num_envs, args.log_base)
        if not success:
            print(f"\n[CURRICULUM STOPPED] Failed at {name}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  CURRICULUM COMPLETE!")
    print(f"  All stages trained successfully.")
    print(f"  Checkpoints in: {args.log_base}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
