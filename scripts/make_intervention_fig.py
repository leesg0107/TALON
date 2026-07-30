"""Intervention-taxonomy figure: outcome composition, depth distributions, and
risk relocation across the three no-retrain grasp-fix families.

Reads logs/graspfix_{off,full,regrasp}_seed1.json (existing data, no sim runs).
Writes data/final_presentation/11_composition_findings/figI_intervention_taxonomy.png

Note: lock_depths is logged once per mission at first lock-in (depth_logged is
never reset on regrasp), so the regrasp depth distribution shows the FIRST
attempt — expected to match baseline; the re-grasp effect appears in outcomes.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/leesg17/Github/soltronev3"
MODES = ["off", "full", "regrasp"]
LABELS = {"off": "baseline\n(off)", "full": "shape + gate\n(ramped close + depth gate)",
          "regrasp": "re-sample\n(detect-and-redraw, K=3)"}

def family(reason):
    if reason.startswith("dock_gated"):
        return "gated (rejected)"
    if reason.startswith("dock") or reason == "box_fell_during_dock":
        return "dock fail"
    if reason.startswith("climb"):
        return "climb fail"
    if "delivery" in reason:
        return "delivery fail"
    return "other"

data = {}
for m in MODES:
    d = json.load(open(f"{ROOT}/logs/graspfix_{m}_seed1.json"))
    n = d["full_success"] + sum(d["fail_reasons"].values())
    fam = {}
    for r, c in d["fail_reasons"].items():
        fam[family(r)] = fam.get(family(r), 0) + c
    data[m] = {"n": n, "success": d["full_success"], "fam": fam,
               "depths": np.array(d["lock_depths"]) * 1000.0}  # mm

FAM_ORDER = ["success", "gated (rejected)", "dock fail", "climb fail", "delivery fail", "other"]
FAM_COLOR = {"success": "#2f9e44", "gated (rejected)": "#adb5bd", "dock fail": "#e8590c",
             "climb fail": "#f2c037", "delivery fail": "#c92a2a", "other": "#dee2e6"}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# --- Panel A: outcome composition (stacked, % of missions) ---
ax = axes[0]
for i, m in enumerate(MODES):
    n = data[m]["n"]
    left = 0.0
    for f in FAM_ORDER:
        c = data[m]["success"] if f == "success" else data[m]["fam"].get(f, 0)
        pct = 100.0 * c / n
        if pct == 0:
            continue
        ax.barh(i, pct, left=left, color=FAM_COLOR[f], edgecolor="white")
        if pct > 4:
            ax.text(left + pct / 2, i, f"{pct:.1f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        left += pct
    ax.text(101, i, f"N={n}", va="center", fontsize=8)
ax.set_yticks(range(len(MODES)))
ax.set_yticklabels([LABELS[m] for m in MODES], fontsize=8)
ax.set_xlabel("% of missions")
ax.set_xlim(0, 112)
ax.invert_yaxis()
handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COLOR[f]) for f in FAM_ORDER]
ax.legend(handles, FAM_ORDER, loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, fontsize=7, framealpha=0.9)
ax.set_title("(a) Mission outcome composition per intervention", fontsize=10)

# --- Panel B: first-lock grasp-depth distributions ---
ax = axes[1]
bins = np.linspace(-45, 5, 26)
for m, color in zip(MODES, ["#495057", "#e8590c", "#1971c2"]):
    dep = data[m]["depths"]
    ax.hist(dep, bins=bins, density=True, histtype="step", lw=2, color=color,
            label=f"{m}: mean {dep.mean():.1f} mm (n={len(dep)})")
ax.axvline(-25, color="k", ls="--", lw=1)
ax.text(-25, ax.get_ylim()[1] * 0.97, " gate threshold (-25 mm)", fontsize=7,
        va="top")
ax.set_xlabel("grasp depth at lock-in, first attempt (mm; more negative = deeper)")
ax.set_ylabel("density")
ax.legend(fontsize=7)
ax.set_title("(b) Ramped close backfires; re-sample draws\nfrom the unchanged first-attempt distribution", fontsize=10)

# --- Panel C: risk relocation (where failure mass lives) + conditional success ---
ax = axes[2]
x = np.arange(len(MODES))
w = 0.35
dock_mass, deliv_mass, e2e, cond = [], [], [], []
for m in MODES:
    n = data[m]["n"]
    dock = data[m]["fam"].get("dock fail", 0) + data[m]["fam"].get("gated (rejected)", 0)
    deliv = data[m]["fam"].get("delivery fail", 0)
    dock_mass.append(100 * dock / n)
    deliv_mass.append(100 * deliv / n)
    e2e.append(100 * data[m]["success"] / n)
    accepted = n - dock
    cond.append(100 * data[m]["success"] / accepted)
ax.bar(x - w / 2, dock_mass, w, color="#e8590c", label="dock-phase failure+rejection mass")
ax.bar(x + w / 2, deliv_mass, w, color="#c92a2a", label="delivery-phase failure mass")
for xi, (dm, vm) in enumerate(zip(dock_mass, deliv_mass)):
    ax.text(xi - w / 2, dm + 1, f"{dm:.1f}", ha="center", fontsize=8)
    ax.text(xi + w / 2, vm + 1, f"{vm:.1f}", ha="center", fontsize=8)
ax2 = ax.twinx()
ax2.plot(x, e2e, "o-", color="#2f9e44", label="end-to-end success")
ax2.plot(x, cond, "s--", color="#1971c2", label="success | handoff accepted")
for xi, (e, c) in enumerate(zip(e2e, cond)):
    ax2.annotate(f"{e:.1f}", (xi, e), textcoords="offset points", xytext=(6, -12),
                 fontsize=8, color="#2f9e44")
    ax2.annotate(f"{c:.1f}", (xi, c), textcoords="offset points", xytext=(6, 6),
                 fontsize=8, color="#1971c2")
ax2.set_ylim(0, 100)
ax2.set_ylabel("success rate (%)")
ax.set_ylim(0, 70)
ax.set_xticks(x)
ax.set_xticklabels([m for m in MODES], fontsize=9)
ax.set_ylabel("failure mass (% of missions)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2)
ax.set_title("(c) Risk relocation: interventions move failure\nmass upstream, none raise end-to-end success", fontsize=10)

fig.tight_layout()
out = f"{ROOT}/data/final_presentation/11_composition_findings/figI_intervention_taxonomy.png"
fig.savefig(out, dpi=160)
print("saved", out)

# console stats for the paper table
for m in MODES:
    n = data[m]["n"]
    acc = n - data[m]["fam"].get("dock fail", 0) - data[m]["fam"].get("gated (rejected)", 0)
    print(f"{m}: N={n} e2e={100*data[m]['success']/n:.1f}% "
          f"cond(accepted)={100*data[m]['success']/acc:.1f}% "
          f"dock_mass={dock_mass[MODES.index(m)]:.1f}% deliv_mass={deliv_mass[MODES.index(m)]:.1f}% "
          f"depth_mean={data[m]['depths'].mean():.1f}mm")
