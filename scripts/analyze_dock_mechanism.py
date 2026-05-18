"""Dock controller mechanism analysis.

Question: Why does the dock controller produce DEEP grasps in some attempts and
SHALLOW grasps in others? The bifurcation happens at contain_hold ≈ 230-280
(plate closure), but the CAUSE must be set EARLIER (contain 100-230).

Strategy:
  1. Group missions by FINAL grasp depth at delivery_entry:
       DEEP: box_offset_y < -0.020
       SHALLOW: box_offset_y > -0.015
  2. For each group, compute mean+std of dock-phase features at each
     contain_hold value, focusing on the predictive window (contain 100-200).
  3. Identify which features differ between groups BEFORE the bifurcation.
  4. Plot side-by-side trajectories.

Outputs:
  logs/dock_mechanism/predictive_features.png — feature trajectories
  logs/dock_mechanism/scatter_predictors.png  — early features vs final depth
  logs/dock_mechanism/summary.txt             — numeric findings
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "logs/diagnose_climb_propagation_raw.npz"
OUT = "logs/dock_mechanism"
os.makedirs(OUT, exist_ok=True)

# 8-D dock trace indices
D_CONT, D_DZ, D_BZ, D_TILT, D_OFFX, D_OFFY, D_OFFZ, D_DIST = range(8)
# Final delivery_entry indices (18-D)
BOX_OFF_Y_FINAL = 15

d = np.load(DATA, allow_pickle=True)
outcomes = d["outcomes"]
dock_traces = d["dock_traces"]
delivery_entry = d["delivery_entry"]
climb_entry = d["climb_entry"]

N = len(outcomes)
print(f"Loaded {N} missions")

# Group missions by climb_entry box_offset_y (same as final, since climb doesn't change it)
has_dock = np.array([tr is not None and len(tr) > 5 for tr in dock_traces])
has_climb = ~np.isnan(climb_entry[:, 0])
valid = has_dock & has_climb

deep_mask = valid & (climb_entry[:, BOX_OFF_Y_FINAL] < -0.020)
shallow_mask = valid & (climb_entry[:, BOX_OFF_Y_FINAL] > -0.015)
mid_mask = valid & ~deep_mask & ~shallow_mask

n_deep = deep_mask.sum()
n_shallow = shallow_mask.sum()
n_mid = mid_mask.sum()
print(f"\nGroups:")
print(f"  DEEP    (offset_y < -0.020): {n_deep}")
print(f"  MID     (-0.020 ≤ y ≤ -0.015): {n_mid}")
print(f"  SHALLOW (offset_y > -0.015): {n_shallow}\n")

# ----------------------------------------------------------------
# Aggregate trajectories by contain_hold value
# ----------------------------------------------------------------
def bin_by_contain(traces_indices, feat_idx, contain_range=range(100, 326, 2)):
    """Return dict {contain_hold: list of feature values across missions}."""
    bins = {c: [] for c in contain_range}
    for i in traces_indices:
        tr = dock_traces[i]
        for row in tr:
            c = int(row[D_CONT])
            if c in bins:
                bins[c].append(row[feat_idx])
    return bins

def trace_stats(bins):
    cs, mu, sd = [], [], []
    for c, vs in sorted(bins.items()):
        if len(vs) >= 3:
            cs.append(c); mu.append(np.mean(vs)); sd.append(np.std(vs))
    return np.array(cs), np.array(mu), np.array(sd)

features = [
    ("drone_z_local (m)", D_DZ),
    ("box_z_local (m)", D_BZ),
    ("drone-box distance (m)", D_DIST),
    ("box_offset_x (strut axis) (m)", D_OFFX),
    ("box_offset_y (plate axis) (m)", D_OFFY),
    ("box_offset_z (m)", D_OFFZ),
    ("drone tilt (deg)", D_TILT),
]

# ----------------------------------------------------------------
# Q. What features differ between DEEP and SHALLOW groups in the
#    PREDICTIVE window (contain 100-200, before bifurcation)?
# ----------------------------------------------------------------
deep_idx = np.where(deep_mask)[0]
shallow_idx = np.where(shallow_mask)[0]

print(f"  Feature differences in PRE-bifurcation window (contain 100-200):")
print(f"  {'Feature':30s}  {'DEEP':>22s}  {'SHALLOW':>22s}  {'Cohen d':>8s}")

with open(f"{OUT}/summary.txt", "w") as fout:
    fout.write(f"DEEP={n_deep}, SHALLOW={n_shallow}, MID={n_mid}\n\n")
    fout.write("Mean ± std of dock-phase features at each contain_hold bucket:\n\n")

    for name, fi in features:
        # Average over contain 100-200 window per mission (one number per mission)
        deep_vals = []
        shallow_vals = []
        for i in deep_idx:
            tr = dock_traces[i]
            mask = (tr[:, D_CONT] >= 100) & (tr[:, D_CONT] <= 200)
            if mask.any():
                deep_vals.append(tr[mask, fi].mean())
        for i in shallow_idx:
            tr = dock_traces[i]
            mask = (tr[:, D_CONT] >= 100) & (tr[:, D_CONT] <= 200)
            if mask.any():
                shallow_vals.append(tr[mask, fi].mean())
        deep_vals = np.array(deep_vals)
        shallow_vals = np.array(shallow_vals)
        if len(deep_vals) > 1 and len(shallow_vals) > 1:
            pooled = np.sqrt((deep_vals.std()**2 + shallow_vals.std()**2) / 2)
            d_eff = (deep_vals.mean() - shallow_vals.mean()) / (pooled + 1e-9)
        else:
            d_eff = float('nan')
        line = (f"  {name:30s}  μ={deep_vals.mean():+.4f}±{deep_vals.std():.4f}    "
                f"μ={shallow_vals.mean():+.4f}±{shallow_vals.std():.4f}    d={d_eff:+.3f}")
        print(line)
        fout.write(line + "\n")

# ----------------------------------------------------------------
# Plot — feature trajectories for each group
# ----------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for ax, (name, fi) in zip(axes, features):
    for mask, label, color in [(deep_mask, "DEEP (good)", "green"),
                                 (shallow_mask, "SHALLOW (bad)", "red"),
                                 (mid_mask, "MID", "gray")]:
        idx = np.where(mask)[0]
        if len(idx) == 0: continue
        bins = bin_by_contain(idx, fi)
        cs, mu, sd = trace_stats(bins)
        if len(cs) == 0: continue
        ax.plot(cs, mu, color=color, label=f"{label} (n={mask.sum()})", linewidth=2)
        ax.fill_between(cs, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.axvline(150, color='gray', linestyle='--', alpha=0.5, label='grasp close trigger')
    ax.axvline(230, color='blue', linestyle='--', alpha=0.5, label='bifurcation start')
    ax.axvline(280, color='black', linestyle='--', alpha=0.5, label='bifurcation end')
    ax.set_xlabel("contain_hold_count")
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# Use last axes for nothing
for ax in axes[len(features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/predictive_features.png", dpi=100)
plt.close()
print(f"\n  Saved: {OUT}/predictive_features.png")

# ----------------------------------------------------------------
# Scatter: early feature value (at contain=150) vs final box_offset_y
# Find: which single feature is most predictive?
# ----------------------------------------------------------------
def get_at_contain(i, contain_target, feat_idx, tolerance=5):
    """Get feature value at contain_hold near target."""
    tr = dock_traces[i]
    mask = np.abs(tr[:, D_CONT] - contain_target) <= tolerance
    if mask.any():
        return tr[mask, feat_idx].mean()
    return float('nan')

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for ax, (name, fi) in zip(axes, features):
    xs, ys = [], []
    colors = []
    for i in np.where(valid)[0]:
        x = get_at_contain(i, 150, fi)  # feature at contain=150 (grasp close moment)
        y = climb_entry[i, BOX_OFF_Y_FINAL]  # final depth
        if not (np.isnan(x) or np.isnan(y)):
            xs.append(x); ys.append(y)
            colors.append('green' if y < -0.020 else ('red' if y > -0.015 else 'gray'))
    if len(xs) > 0:
        ax.scatter(xs, ys, c=colors, alpha=0.4, s=12)
        # Compute correlation
        xs_a, ys_a = np.array(xs), np.array(ys)
        if xs_a.std() > 1e-6:
            corr = np.corrcoef(xs_a, ys_a)[0, 1]
            ax.set_title(f"{name} @ contain=150 (corr={corr:+.3f})", fontsize=9)
        else:
            ax.set_title(f"{name} @ contain=150", fontsize=9)
        ax.axhline(-0.020, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(-0.015, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(name)
    ax.set_ylabel("final box_offset_y (m)")
    ax.grid(alpha=0.3)

# Hide unused axes
for ax in axes[len(features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/scatter_predictors.png", dpi=100)
plt.close()
print(f"  Saved: {OUT}/scatter_predictors.png")
print(f"  Saved: {OUT}/summary.txt")
print("\nDone.")
