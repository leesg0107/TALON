"""Offline analyzer for diagnose_climb_propagation_raw.npz.

Answers three questions:
  Q3. Does shallow grasp form in DOCK, or during CLIMB?
      → compare box_offset_y at climb_entry (DOCK exit) vs delivery_entry (CLIMB exit).
        if same → DOCK origin. if drift during climb → CLIMB origin.

  Q4. WHEN during DOCK does the box become shallow?
      → plot box_offset_y vs contain_hold for failed vs success missions.

  Q5. Are box_dropped and too_tilted_delivery the same mechanism?
      → separate per-outcome feature comparison + Cohen's d.

Outputs:
  logs/diagnose_propagation/q3_dock_vs_delivery.png
  logs/diagnose_propagation/q4_dock_trajectory.png
  logs/diagnose_propagation/q5_mechanism_split.png
  logs/diagnose_propagation/summary.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "logs/diagnose_climb_propagation_raw.npz"
OUT = "logs/diagnose_propagation"
os.makedirs(OUT, exist_ok=True)

# 18-D snapshot indices
TILT, ANG_VEL = 6, 7
VZ = 5
BOX_OFF_X, BOX_OFF_Y, BOX_OFF_Z = 14, 15, 16
BOX_DIST = 17

# 8-D dock trace indices: contain_hold, drone_z, box_z, tilt, off_x, off_y, off_z, dist
D_CONT, D_DZ, D_BZ, D_TILT, D_OFFX, D_OFFY, D_OFFZ, D_DIST = range(8)

d = np.load(DATA, allow_pickle=True)
outcomes = d["outcomes"]
dock_traces = d["dock_traces"]
climb_entry = d["climb_entry"]
delivery_entry = d["delivery_entry"]

N = len(outcomes)
print(f"Loaded {N} missions")
unique, counts = np.unique(outcomes, return_counts=True)
print(f"Outcomes: {dict(zip(unique, counts))}\n")

mask_success = (outcomes == "full_success")
mask_box_dropped = (outcomes == "box_dropped_delivery")
mask_too_tilted = np.array([str(o).startswith("too_tilted") for o in outcomes])
mask_climb_fail = np.array(["climb" in str(o) for o in outcomes])

# ----------------------------------------------------------------
# Q3: DOCK origin vs CLIMB drift
# ----------------------------------------------------------------
print("=" * 60)
print("  Q3. Shallow grasp origin — DOCK or CLIMB?")
print("=" * 60)
has_both = ~np.isnan(climb_entry[:, 0]) & ~np.isnan(delivery_entry[:, 0])

print(f"\n  box_offset_y (negative = box deeper in gripper):")
print(f"  {'Outcome':30s}  {'@DOCK exit':>20s}  {'@DELIVERY entry':>20s}  {'Δ (climb drift)':>18s}")

for mask, label in [(mask_success, "success"),
                     (mask_box_dropped, "box_dropped"),
                     (mask_too_tilted, "too_tilted")]:
    sel = mask & has_both
    if sel.sum() == 0: continue
    de = climb_entry[sel, BOX_OFF_Y]
    le = delivery_entry[sel, BOX_OFF_Y]
    drift = le - de
    print(f"  {label:30s}  μ={de.mean():+0.4f}±{de.std():.4f}    "
          f"μ={le.mean():+0.4f}±{le.std():.4f}    μ={drift.mean():+0.4f}±{drift.std():.4f}")

print(f"\n  Interpretation:")
print(f"  - If 'Δ climb drift' near 0 → box stays put during climb → DOCK is origin")
print(f"  - If Δ negative→positive  → box becomes shallower during climb → CLIMB is origin")

# Plot: scatter of climb_entry box_offset_y vs delivery_entry box_offset_y, colored by outcome
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for mask, label, color in [(mask_success, "success", "green"),
                             (mask_box_dropped, "box_dropped", "red"),
                             (mask_too_tilted, "too_tilted", "orange")]:
    sel = mask & has_both
    if sel.sum() == 0: continue
    ax.scatter(climb_entry[sel, BOX_OFF_Y], delivery_entry[sel, BOX_OFF_Y],
               alpha=0.4, label=f"{label} (n={sel.sum()})", color=color, s=20)
lim_min = min(climb_entry[has_both, BOX_OFF_Y].min(), delivery_entry[has_both, BOX_OFF_Y].min())
lim_max = max(climb_entry[has_both, BOX_OFF_Y].max(), delivery_entry[has_both, BOX_OFF_Y].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, label='y=x (no drift)')
ax.set_xlabel("box_offset_y at DOCK exit (m)")
ax.set_ylabel("box_offset_y at DELIVERY entry (m)")
ax.set_title("Q3: Grasp depth propagation through CLIMB")
ax.legend()
ax.grid(alpha=0.3)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{OUT}/q3_dock_vs_delivery.png", dpi=100)
plt.close()
print(f"  Saved: {OUT}/q3_dock_vs_delivery.png")

# ----------------------------------------------------------------
# Q4: WHEN during DOCK does shallowing occur?
# ----------------------------------------------------------------
print(f"\n{'='*60}")
print("  Q4. WHEN during DOCK does shallowing happen?")
print("=" * 60)

# Bin dock traces by contain_hold count, plot mean box_offset_y trajectory
# for each outcome group.
def collect_dock_trace_by_contain(mask):
    """Return dict {contain_hold: list of box_offset_y values across missions}."""
    bins = {}
    for i in np.where(mask)[0]:
        tr = dock_traces[i]
        if tr is None or len(tr) < 5: continue
        for row in tr:
            c = int(row[D_CONT])
            bins.setdefault(c, []).append(row[D_OFFY])
    return bins

def trace_stats(bins, contain_range=range(100, 326, 5)):
    cs, means, stds, ns = [], [], [], []
    for c in contain_range:
        vs = bins.get(c, [])
        if len(vs) >= 3:
            cs.append(c); means.append(np.mean(vs))
            stds.append(np.std(vs)); ns.append(len(vs))
    return np.array(cs), np.array(means), np.array(stds), np.array(ns)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for mask, label, color in [(mask_success, "success", "green"),
                             (mask_box_dropped, "box_dropped", "red"),
                             (mask_too_tilted, "too_tilted", "orange"),
                             (mask_climb_fail, "climb_failed", "purple")]:
    bins = collect_dock_trace_by_contain(mask)
    cs, mu, sd, ns = trace_stats(bins)
    if len(cs) == 0: continue
    ax.plot(cs, mu, color=color, label=f"{label} (n_traces={mask.sum()})", linewidth=2)
    ax.fill_between(cs, mu - sd, mu + sd, color=color, alpha=0.15)

ax.axvline(150, color='gray', linestyle='--', alpha=0.5, label='grasp close trigger')
ax.axvline(325, color='black', linestyle='--', alpha=0.5, label='dock-exit / climb start')
ax.set_xlabel("contain_hold_count (dock progress)")
ax.set_ylabel("box_offset_y (m, negative = deeper)")
ax.set_title("Q4: When does box_offset_y diverge?")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for mask, label, color in [(mask_success, "success", "green"),
                             (mask_box_dropped, "box_dropped", "red"),
                             (mask_too_tilted, "too_tilted", "orange"),
                             (mask_climb_fail, "climb_failed", "purple")]:
    bins_tilt = {}
    for i in np.where(mask)[0]:
        tr = dock_traces[i]
        if tr is None or len(tr) < 5: continue
        for row in tr:
            c = int(row[D_CONT])
            bins_tilt.setdefault(c, []).append(row[D_TILT])
    cs, mu, sd, ns = [], [], [], []
    for c in range(100, 326, 5):
        vs = bins_tilt.get(c, [])
        if len(vs) >= 3:
            cs.append(c); mu.append(np.mean(vs))
            sd.append(np.std(vs)); ns.append(len(vs))
    cs, mu, sd = np.array(cs), np.array(mu), np.array(sd)
    if len(cs) == 0: continue
    ax.plot(cs, mu, color=color, label=label, linewidth=2)
    ax.fill_between(cs, mu - sd, mu + sd, color=color, alpha=0.15)

ax.axvline(150, color='gray', linestyle='--', alpha=0.5)
ax.axvline(325, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel("contain_hold_count")
ax.set_ylabel("drone tilt (deg)")
ax.set_title("Q4b: Drone tilt evolution during DOCK")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/q4_dock_trajectory.png", dpi=100)
plt.close()
print(f"  Saved: {OUT}/q4_dock_trajectory.png")

# ----------------------------------------------------------------
# Q5: box_dropped vs too_tilted — same mechanism?
# ----------------------------------------------------------------
print(f"\n{'='*60}")
print("  Q5. box_dropped vs too_tilted_delivery — same mechanism?")
print("=" * 60)

def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2: return float('nan')
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled < 1e-6: return 0.0
    return (a.mean() - b.mean()) / pooled

features = [
    ("tilt_deg @climb_entry", climb_entry, TILT),
    ("ang_vel @climb_entry", climb_entry, ANG_VEL),
    ("box_offset_y @climb_entry", climb_entry, BOX_OFF_Y),
    ("tilt_deg @delivery_entry", delivery_entry, TILT),
    ("ang_vel @delivery_entry", delivery_entry, ANG_VEL),
    ("vel_z @delivery_entry", delivery_entry, VZ),
    ("box_offset_y @delivery_entry", delivery_entry, BOX_OFF_Y),
]

print(f"\n  Cohen's d effect sizes:")
print(f"  {'Feature':35s}  {'success vs box_dropped':>22s}  {'success vs too_tilted':>22s}")
for name, arr, fi in features:
    s_v = arr[mask_success & ~np.isnan(arr[:, 0]), fi]
    b_v = arr[mask_box_dropped & ~np.isnan(arr[:, 0]), fi]
    t_v = arr[mask_too_tilted & ~np.isnan(arr[:, 0]), fi]
    if len(s_v) > 0 and len(b_v) > 0 and len(t_v) > 0:
        d_sb = cohens_d(s_v, b_v)
        d_st = cohens_d(s_v, t_v)
        print(f"  {name:35s}  d={d_sb:>+6.3f}              d={d_st:>+6.3f}")

# Plot side-by-side histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plot_features = [
    ("tilt @climb_entry", climb_entry, TILT),
    ("ang_vel @climb_entry", climb_entry, ANG_VEL),
    ("box_offset_y @climb_entry", climb_entry, BOX_OFF_Y),
    ("vel_z @climb_entry", climb_entry, VZ),
    ("tilt @delivery_entry", delivery_entry, TILT),
    ("ang_vel @delivery_entry", delivery_entry, ANG_VEL),
    ("box_offset_y @delivery_entry", delivery_entry, BOX_OFF_Y),
    ("vel_z @delivery_entry", delivery_entry, VZ),
]
for ax, (name, arr, fi) in zip(axes.flatten(), plot_features):
    s_v = arr[mask_success & ~np.isnan(arr[:, 0]), fi]
    b_v = arr[mask_box_dropped & ~np.isnan(arr[:, 0]), fi]
    t_v = arr[mask_too_tilted & ~np.isnan(arr[:, 0]), fi]
    if len(s_v): ax.hist(s_v, bins=20, alpha=0.5, label='success', color='green', density=True)
    if len(b_v): ax.hist(b_v, bins=20, alpha=0.5, label='box_dropped', color='red', density=True)
    if len(t_v): ax.hist(t_v, bins=20, alpha=0.5, label='too_tilted', color='orange', density=True)
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/q5_mechanism_split.png", dpi=100)
plt.close()
print(f"\n  Saved: {OUT}/q5_mechanism_split.png")
print(f"\nDone.")
