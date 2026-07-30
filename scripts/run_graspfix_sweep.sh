#!/usr/bin/env bash
# Grasp-fix ablation: seed-pairing check + full sweep, run SEQUENTIALLY.
# One GPU -> one Isaac instance at a time (concurrent runs would OOM/contend).
# Launch once and walk away:  bash scripts/run_graspfix_sweep.sh 2>&1 | tee logs/sweep_all.log
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-1 2 3 4}"
MODES="${MODES:-nogate off gate full}"
N="${N:-500}"   # drain phase makes this unbiased; 500 for tight CI on absolute success

if [ "${SKIP_PAIR:-0}" != "1" ]; then
  echo "############ Step 0: seed-pairing reproducibility check ############"
  python scripts/eval_mission_headless.py --grasp-fix off --seed 1 --target-missions 128 2>&1 | tee logs/sf_pair_a.log
  python scripts/eval_mission_headless.py --grasp-fix off --seed 1 --target-missions 128 2>&1 | tee logs/sf_pair_b.log
  echo "==== PAIRING CHECK (these two should be ~identical) ===="
  grep -H "Full success" logs/sf_pair_a.log logs/sf_pair_b.log
  echo "  (if they differ a lot, seeding does not pair -> tell Claude; depth result is still valid)"
else
  echo "############ Step 0 skipped (SKIP_PAIR=1; pairing already confirmed) ############"
fi

echo "############ Step 1: full sweep (${MODES} x seeds ${SEEDS}, N=${N}) ############"
for s in $SEEDS; do
  for m in $MODES; do
    echo "---- grasp-fix=$m seed=$s ----"
    python scripts/eval_mission_headless.py --grasp-fix "$m" --seed "$s" --target-missions "$N" \
      2>&1 | tee "logs/graspfix_run_${m}_${s}.log"
  done
done

echo "############ Step 2: analysis ############"
python scripts/analyze_graspfix.py
echo "DONE -> figs in data/final_presentation/10_prescription_validation/ (fig7 depth, fig8 paired success)"
