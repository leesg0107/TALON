#!/bin/bash
# Automated Stage 1→2→3 training pipeline (gripper-centric)
# Usage: ./train_all.sh

set -e  # Stop on any error

echo "=== Stage 1: Basic Flight (gripper-centric) ==="
./run.sh train.py --stage 1 --num_envs 4096 --max_steps 500000000 \
    --log_dir logs/stage1_gripper

# Find latest best checkpoint
STAGE1_CKPT=$(ls -t runs/*PPO/checkpoints/best_agent.pt | head -1)
echo "=== Stage 1 done. Checkpoint: $STAGE1_CKPT ==="

echo "=== Stage 2: Precision Approach (gripper-centric) ==="
./run.sh train.py --stage 2 --num_envs 4096 --max_steps 500000000 \
    --checkpoint $STAGE1_CKPT \
    --log_dir logs/stage2_gripper

STAGE2_CKPT=$(ls -t runs/*PPO/checkpoints/best_agent.pt | head -1)
echo "=== Stage 2 done. Checkpoint: $STAGE2_CKPT ==="

echo "=== Stage 3a: Grasping ==="
./run.sh train.py --stage 3 --substage a --num_envs 4096 --max_steps 500000000 \
    --checkpoint $STAGE2_CKPT \
    --log_dir logs/stage3a_gripper

echo "=== All stages complete ==="
