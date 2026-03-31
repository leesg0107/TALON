#!/bin/bash
# Stage 2→3 training pipeline (10cm box, friction 2.0, gripper-centric)
set -e

STAGE1_CKPT="logs/stage1_gripper/final_agent.pt"

echo "=== Stage 2: Precision Approach (10cm box) ==="
./run.sh train.py --stage 2 --num_envs 4096 --max_steps 500000000 \
    --checkpoint $STAGE1_CKPT \
    --log_dir logs/stage2_10cm

STAGE2_CKPT=$(ls -t runs/*PPO/checkpoints/best_agent.pt | head -1)
echo "=== Stage 2 done. Checkpoint: $STAGE2_CKPT ==="

echo "=== Stage 3a: Grasping (10cm box) ==="
./run.sh train.py --stage 3 --substage a --num_envs 4096 --max_steps 500000000 \
    --checkpoint $STAGE2_CKPT \
    --log_dir logs/stage3a_10cm

echo "=== All stages complete ==="
