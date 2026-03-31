#!/bin/bash
# Wrapper to run soltronev3 scripts via Isaac Lab's launcher.
# Usage:
#   ./run.sh train.py --stage 1 --num_envs 4096 --headless
#   ./run.sh play.py --stage 1 --checkpoint logs/stage1/final_agent.pt

ISAACLAB_DIR="/home/leesg17/Applications/IsaacLab"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Ensure correct conda env (Python 3.11 for Isaac Sim v5.1)
source /home/leesg17/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab311

SCRIPT="$1"
shift

exec "${ISAACLAB_DIR}/isaaclab.sh" -p "${PROJECT_DIR}/${SCRIPT}" "$@"
