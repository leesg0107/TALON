"""
Analyze seed-paired grasp-fix ablation (Phase B / Arm C, no RL retrain).

Reads logs/graspfix_{off,gate,full}_seed{N}.json produced by eval_mission_headless.py
and tests, seed-paired:
  ①′ ramped close:  does mode 'full' produce DEEPER grasps than 'off'?  (lock-depth DV)
  ②  depth gate:    does gating reduce downstream delivery failures, and at what
                    rejection (false-handoff) cost?
  overall:          seed-paired full_success delta (Wilcoxon signed-rank), CI.

Run AFTER collecting >=4 seeds per mode, e.g.:
  for s in 1 2 3 4 5 6 7 8; do
    for m in off gate full; do
      python scripts/eval_mission_headless.py --grasp-fix $m --seed $s --target-missions 256 \
        2>&1 | tee logs/graspfix_run_${m}_${s}.log
    done
  done
  python scripts/analyze_graspfix.py
"""
import os, glob, json
import numpy as np
import scipy.stats as ss
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "final_presentation", "10_prescription_validation")
os.makedirs(OUT, exist_ok=True)
MODES = ["nogate", "off", "gate", "full"]
COL = {"nogate": "#d62828", "off": "#999999", "gate": "#457b9d", "full": "#2a9d8f"}

def deliv_fail(fr):  # downstream (composition) failure count
    return sum(v for k, v in fr.items()
               if k in ("box_dropped_delivery", "too_tilted_delivery",
                        "box_dropped_hover_stab", "too_tilted_hover_stab"))
def gated_reject(fr):
    return sum(v for k, v in fr.items() if k.startswith("dock_gated"))
def climb_fail(fr):   # tilt-driven failure class (recoverable in principle)
    return sum(v for k, v in fr.items() if "climb" in k)
def dock_timeout(fr):
    return sum(v for k, v in fr.items() if k.startswith("dock_timeout"))

# ---- load ----
runs = {m: {} for m in MODES}   # mode -> seed -> record
for f in glob.glob(os.path.join(ROOT, "logs", "graspfix_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    m, s = d.get("mode"), d.get("seed")
    if m in runs:
        runs[m][s] = d
for m in MODES:
    print(f"mode {m:5s}: seeds {sorted(k for k in runs[m] if k is not None)}")
if not any(runs[m] for m in MODES):
    print("\nNo logs/graspfix_*.json yet — run the evals first (see header).")
    raise SystemExit

# ---- ①′ lock-depth distributions (pooled) ----
print("\n--- ①′ ramped close: lock-in grasp depth (more negative = deeper) ---")
depth_pool = {}
for m in MODES:
    pool = np.array([x for r in runs[m].values() for x in r.get("lock_depths", [])])
    depth_pool[m] = pool
    if len(pool):
        print(f"  {m:5s} n={len(pool):4d}  mean={pool.mean()*1000:6.1f}mm  "
              f"deep(<-25mm)={float((pool<-0.025).mean())*100:4.1f}%  "
              f"shallow(>-20mm)={float((pool>-0.020).mean())*100:4.1f}%")
if len(depth_pool.get("off", [])) and len(depth_pool.get("full", [])):
    u, p = ss.mannwhitneyu(depth_pool["full"], depth_pool["off"], alternative="less")
    print(f"  Mann-Whitney full deeper than off: p={p:.2e}  "
          f"(Δmean={ (depth_pool['full'].mean()-depth_pool['off'].mean())*1000:+.1f}mm)")

# ---- seed-paired success + class-specific failure ----
print("\n--- seed-paired comparison (vs off) ---")
seeds = sorted(set(s for m in MODES for s in runs[m] if s is not None))
def rate(d, fn):
    n = d.get("total", 0) + d.get("full_success", 0)
    return fn(d) / max(n, 1)
def succ_rate(d):
    n = d.get("total", 0) + d.get("full_success", 0)
    return d.get("full_success", 0) / max(n, 1)

COMPARE = ("nogate", "gate", "full")  # all compared vs off
paired = {m: {"succ_d": [], "deliv_d": [], "climb_d": [], "to_d": [], "rej": []} for m in COMPARE}
for s in seeds:
    if s not in runs.get("off", {}):
        continue
    o = runs["off"][s]
    for m in COMPARE:
        if s not in runs.get(m, {}):
            continue
        d = runs[m][s]
        paired[m]["succ_d"].append(succ_rate(d) - succ_rate(o))
        paired[m]["deliv_d"].append(rate(d, deliv_fail) - rate(o, deliv_fail))
        paired[m]["climb_d"].append(rate(d, climb_fail) - rate(o, climb_fail))
        paired[m]["to_d"].append(rate(d, dock_timeout) - rate(o, dock_timeout))
        paired[m]["rej"].append(rate(d, gated_reject))
print("  (Δ = mode − off, seed-paired.  nogate−off isolates the ATTITUDE/tilt gate:")
print("   if the tilt gate truly helps, nogate should show +Δclimb_fail and −Δsuccess vs off)")
for m in COMPARE:
    sd = np.array(paired[m]["succ_d"]); dd = np.array(paired[m]["deliv_d"])
    cd = np.array(paired[m]["climb_d"]); td = np.array(paired[m]["to_d"]); rj = np.array(paired[m]["rej"])
    if len(sd) < 2:
        print(f"  {m}: need >=2 paired seeds (have {len(sd)})"); continue
    wp = ss.wilcoxon(sd).pvalue if np.any(sd) else 1.0
    print(f"  {m:6s} (n={len(sd)}): Δsuccess={sd.mean()*100:+.1f}pp "
          f"[{np.percentile(sd,2.5)*100:+.1f},{np.percentile(sd,97.5)*100:+.1f}] Wilcoxon p={wp:.3f}")
    print(f"          Δclimb_fail={cd.mean()*100:+.1f}pp  Δdelivery_fail={dd.mean()*100:+.1f}pp  "
          f"Δdock_timeout={td.mean()*100:+.1f}pp  gated_reject={rj.mean()*100:.1f}%")

# ============================================================ figures
# fig A — lock depth distributions
have = [m for m in MODES if len(depth_pool.get(m, []))]
if have:
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in have:
        ax.hist(depth_pool[m]*1000, bins=30, density=True, alpha=0.5,
                color=COL[m], label=f"{m} (n={len(depth_pool[m])}, deep {float((depth_pool[m]<-0.025).mean())*100:.0f}%)")
    ax.axvline(-25, color="k", ls="--", lw=1); ax.text(-25, ax.get_ylim()[1]*0.9, " deep threshold", fontsize=8)
    ax.set_xlabel("lock-in grasp depth box_offset_y (mm)  — more negative = deeper")
    ax.set_ylabel("density"); ax.set_title("")  # title lives on the slide
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_graspfix_depth.png")); plt.close(fig)
    print("\nwrote fig7_graspfix_depth.png")

# fig B — seed-paired success delta
if any(len(paired[m]["succ_d"]) >= 2 for m in ("gate", "full")):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    xs = []
    for i, m in enumerate(("gate", "full")):
        sd = np.array(paired[m]["succ_d"]) * 100
        if len(sd) < 1: continue
        ax.scatter(np.full(len(sd), i) + np.linspace(-0.05, 0.05, len(sd)), sd, color=COL[m], zorder=3)
        ax.bar(i, sd.mean(), color=COL[m], alpha=0.4, width=0.5)
        xs.append((i, m))
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks([i for i, _ in xs]); ax.set_xticklabels([m for _, m in xs])
    ax.set_ylabel("Δ full-success vs off (pp, seed-paired)")
    ax.set_title("Seed-paired success change (each dot = one seed)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_graspfix_paired_success.png")); plt.close(fig)
    print("wrote fig8_graspfix_paired_success.png")
