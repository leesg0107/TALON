#!/bin/bash
# Stage 1→2→3 training: gripping center reference (-0.08), gripper locked, align focus
set -e

echo "=== Stage 1: Basic Flight (grip center ref) ==="
./run.sh train.py --stage 1 --num_envs 4096 --max_steps 500000000 \
    --log_dir logs/stage1_grip_center

STAGE1_CKPT=$(ls -t runs/*PPO/checkpoints/best_agent.pt | head -1)
echo "=== Stage 1 done. Checkpoint: $STAGE1_CKPT ==="

echo "=== Stage 2: Precision Approach (align focus) ==="
./run.sh train.py --stage 2 --num_envs 4096 --max_steps 500000000 \
    --checkpoint $STAGE1_CKPT \
    --log_dir logs/stage2_grip_center

STAGE2_CKPT=$(ls -t runs/*PPO/checkpoints/best_agent.pt | head -1)
echo "=== Stage 2 done. Checkpoint: $STAGE2_CKPT ==="

echo "=== Stage 3: Align Max (gripper locked open) ==="
./run.sh train.py --stage 3 --substage a --num_envs 4096 --max_steps 500000000 \
    --checkpoint $STAGE2_CKPT \
    --log_dir logs/stage3a_grip_center

echo "=== All stages complete ==="
