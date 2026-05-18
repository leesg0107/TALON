"""Offline analysis for diagnose_climb_delivery_raw.npz.

Answers two questions:
  Q1. Why does PID-controlled CLIMB fail?
      → Compare climb trajectories: climb_failed vs success.
        Look at when z/vel_z/tilt diverge, and what climb-entry features predict failure.
  Q2. When CLIMB succeeds, is residual instability the cause of DELIVERY failure?
      → Compare delivery-entry features (drone tilt, ang_vel, box offset) for
        success vs box_dropped_delivery vs too_tilted_delivery.

Outputs:
  logs/diagnose_climb_delivery/q1_climb_trajectories.png  — sample climb fail vs success
  logs/diagnose_climb_delivery/q1_entry_predictors.png    — climb_entry distributions
  logs/diagnose_climb_delivery/q2_delivery_entry.png      — delivery_entry split by outcome
  logs/diagnose_climb_delivery/summary.txt                — numeric findings

Run:
    python scripts/analyze_climb_delivery.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "logs/diagnose_climb_delivery_raw.npz"
OUT_DIR = "logs/diagnose_climb_delivery"
os.makedirs(OUT_DIR, exist_ok=True)

# Feature layout (18-D snapshot, matches snapshot_state in diagnostic):
# 0-2: drone_pos_local (x,y,z)
# 3-5: drone_vel_world (vx,vy,vz)
# 6: drone_tilt_deg
# 7: drone_ang_vel_body_norm
# 8-10: box_pos_local
# 11-13: box_vel_world
# 14-16: box_offset_in_gripper_body
# 17: drone_box_dist
DRONE_X, DRONE_Y, DRONE_Z = 0, 1, 2
DRONE_VX, DRONE_VY, DRONE_VZ = 3, 4, 5
TILT = 6
ANG_VEL = 7
BOX_X, BOX_Y, BOX_Z = 8, 9, 10
BOX_VX, BOX_VY, BOX_VZ = 11, 12, 13
BOX_OFF_X, BOX_OFF_Y, BOX_OFF_Z = 14, 15, 16
BOX_DIST = 17

# Climb trace features (6-D):
# 0: drone_z_local, 1: drone_vel_z, 2: tilt_deg, 3: ang_vel_norm, 4: box_z_local, 5: drone_box_dist
T_Z, T_VZ, T_TILT, T_ANGV, T_BOXZ, T_DIST = range(6)

# ----------------------------------------------------------------
# Load
# ----------------------------------------------------------------
d = np.load(DATA, allow_pickle=True)
outcomes = d["outcomes"]
climb_entry = d["climb_entry"]         # (N, 18)
delivery_entry = d["delivery_entry"]   # (N, 18)
climb_traces = d["climb_traces"]        # array of (T_i, 6)
dr_samples = d["dr_samples"]            # (N, 2)

N = len(outcomes)
print(f"Loaded {N} missions")
print(f"Outcomes: {dict(zip(*np.unique(outcomes, return_counts=True)))}\n")

# Mission categories
mask_success = (outcomes == "full_success")
mask_climb_fail = np.array(["climb" in str(o) for o in outcomes])
mask_box_dropped = (outcomes == "box_dropped_delivery")
mask_too_tilted = np.array([str(o).startswith("too_tilted") for o in outcomes])
mask_reached_delivery = ~np.isnan(delivery_entry[:, 0])

n_climb_fail = mask_climb_fail.sum()
n_success = mask_success.sum()
n_box_dropped = mask_box_dropped.sum()
n_too_tilted = mask_too_tilted.sum()
n_reached_delivery = mask_reached_delivery.sum()

# ----------------------------------------------------------------
# Q1: CLIMB failure analysis
# ----------------------------------------------------------------
print("=" * 60)
print("  Q1. Why does CLIMB fail?")
print("=" * 60)
print(f"  CLIMB failures: {n_climb_fail}/{N} ({100*n_climb_fail/N:.1f}%)")

# Q1.a: Compare climb-entry features (climb_fail vs success)
print(f"\n  CLIMB-entry features (DOCK→CLIMB transition):")
feat_names = [
    ("drone_tilt_deg", TILT),
    ("ang_vel_norm", ANG_VEL),
    ("drone_vel_z", DRONE_VZ),
    ("drone_z_local", DRONE_Z),
    ("box_offset_x (strut axis)", BOX_OFF_X),
    ("box_offset_y (plate axis)", BOX_OFF_Y),
    ("box_offset_z", BOX_OFF_Z),
    ("box_dist", BOX_DIST),
]

# Only consider missions that reached climb (climb_entry not NaN)
has_climb = ~np.isnan(climb_entry[:, 0])
cf_climb = climb_entry[mask_climb_fail & has_climb]
ok_climb = climb_entry[mask_success & has_climb]

print(f"  {'Feature':25s}  {'climb_fail':>20s}  {'success':>20s}")
for name, fi in feat_names:
    cf_v = cf_climb[:, fi]
    ok_v = ok_climb[:, fi]
    print(f"  {name:25s}  μ={cf_v.mean():>+6.3f}±{cf_v.std():>5.3f}    μ={ok_v.mean():>+6.3f}±{ok_v.std():>5.3f}")

# Q1.b: DR sample correlation
print(f"\n  DR sample correlation:")
cf_dr = dr_samples[mask_climb_fail]
ok_dr = dr_samples[mask_success]
print(f"  {'mass_scale':25s}  μ={cf_dr[:, 0].mean():>+6.3f}±{cf_dr[:, 0].std():>5.3f}    μ={ok_dr[:, 0].mean():>+6.3f}±{ok_dr[:, 0].std():>5.3f}")
print(f"  {'payload_mass':25s}  μ={cf_dr[:, 1].mean():>+6.3f}±{cf_dr[:, 1].std():>5.3f}    μ={ok_dr[:, 1].mean():>+6.3f}±{ok_dr[:, 1].std():>5.3f}")

# Q1.c: Climb trajectory shape (sample plots)
print(f"\n  Generating climb trajectory plots...")
cf_indices = np.where(mask_climb_fail)[0]
ok_indices = np.where(mask_success)[0]

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
# Top 3 longest fail traces (most informative)
cf_traces_sorted = sorted([(i, climb_traces[i]) for i in cf_indices
                          if climb_traces[i] is not None and len(climb_traces[i]) > 10],
                         key=lambda x: -len(x[1]))[:3]
for i, (idx, tr) in enumerate(cf_traces_sorted):
    t = np.arange(len(tr)) / 150.0
    axes[i, 0].plot(t, tr[:, T_Z], 'r-', label='drone z')
    axes[i, 0].plot(t, tr[:, T_BOXZ], 'r--', alpha=0.6, label='box z')
    axes[i, 0].axhline(1.0, color='gray', linestyle=':', alpha=0.5, label='climb target')
    axes[i, 0].axhline(0.30, color='red', linestyle=':', alpha=0.5, label='ground crash')
    axes[i, 0].set_ylabel('altitude (m)')
    axes[i, 0].set_title(f"CLIMB FAIL #{i+1} (mission {idx})")
    axes[i, 0].legend(fontsize=7); axes[i, 0].grid(alpha=0.3)

    axes[i, 1].plot(t, tr[:, T_TILT], 'orange', label='tilt (deg)')
    ax2 = axes[i, 1].twinx()
    ax2.plot(t, tr[:, T_VZ], 'b-', alpha=0.5, label='vel_z (m/s)')
    ax2.set_ylabel('vel_z (m/s)', color='b')
    axes[i, 1].set_ylabel('tilt (deg)', color='orange')
    axes[i, 1].legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    axes[i, 1].grid(alpha=0.3)

# Sample success trace
if len(ok_indices) > 0:
    sample_ok = next((i for i in ok_indices if climb_traces[i] is not None and len(climb_traces[i]) > 5), None)
    if sample_ok is not None:
        tr = climb_traces[sample_ok]
        t = np.arange(len(tr)) / 150.0
        axes[3, 0].plot(t, tr[:, T_Z], 'g-', label='drone z')
        axes[3, 0].plot(t, tr[:, T_BOXZ], 'g--', alpha=0.6, label='box z')
        axes[3, 0].axhline(1.0, color='gray', linestyle=':', alpha=0.5)
        axes[3, 0].set_ylabel('altitude (m)'); axes[3, 0].set_xlabel('time (s)')
        axes[3, 0].set_title(f"SUCCESS (mission {sample_ok})")
        axes[3, 0].legend(fontsize=7); axes[3, 0].grid(alpha=0.3)

        axes[3, 1].plot(t, tr[:, T_TILT], 'orange')
        ax2 = axes[3, 1].twinx()
        ax2.plot(t, tr[:, T_VZ], 'b-', alpha=0.5)
        axes[3, 1].set_ylabel('tilt (deg)', color='orange')
        axes[3, 1].set_xlabel('time (s)')
        ax2.set_ylabel('vel_z', color='b')
        axes[3, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q1_climb_trajectories.png", dpi=100); plt.close()
print(f"    Saved: {OUT_DIR}/q1_climb_trajectories.png")

# Q1.d: Climb-entry distribution histograms
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, (name, fi) in zip(axes.flatten(), feat_names):
    if (mask_success & has_climb).sum() > 0:
        ax.hist(climb_entry[mask_success & has_climb, fi], bins=20, alpha=0.5, label='success', color='blue', density=True)
    if (mask_climb_fail & has_climb).sum() > 0:
        ax.hist(climb_entry[mask_climb_fail & has_climb, fi], bins=20, alpha=0.5, label='climb_fail', color='red', density=True)
    ax.set_title(name, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q1_entry_predictors.png", dpi=100); plt.close()
print(f"    Saved: {OUT_DIR}/q1_entry_predictors.png")

# ----------------------------------------------------------------
# Q2: CLIMB→DELIVERY propagation
# ----------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Q2. Does CLIMB-exit instability cause DELIVERY failure?")
print("=" * 60)
print(f"  Missions that reached delivery: {n_reached_delivery}")
print(f"  → success: {n_success}, box_dropped: {n_box_dropped}, too_tilted: {n_too_tilted}\n")

print(f"  DELIVERY-entry features (CLIMB→DELIVERY transition):")
print(f"  {'Feature':25s}  {'success':>22s}  {'box_dropped':>22s}  {'too_tilted':>22s}")
for name, fi in feat_names:
    s_v = delivery_entry[mask_success & mask_reached_delivery, fi]
    b_v = delivery_entry[mask_box_dropped & mask_reached_delivery, fi]
    t_v = delivery_entry[mask_too_tilted & mask_reached_delivery, fi]
    print(f"  {name:25s}  μ={s_v.mean():>+6.3f}±{s_v.std():>5.3f}    μ={b_v.mean():>+6.3f}±{b_v.std():>5.3f}    "
          f"μ={t_v.mean():>+6.3f}±{t_v.std():>5.3f}")

# Q2 visualization: delivery-entry histograms by outcome
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, (name, fi) in zip(axes.flatten(), feat_names):
    ax.hist(delivery_entry[mask_success & mask_reached_delivery, fi], bins=20, alpha=0.5, label='success', color='green', density=True)
    if mask_box_dropped.sum() > 0:
        ax.hist(delivery_entry[mask_box_dropped & mask_reached_delivery, fi], bins=20, alpha=0.5, label='box_dropped', color='red', density=True)
    if mask_too_tilted.sum() > 0:
        ax.hist(delivery_entry[mask_too_tilted & mask_reached_delivery, fi], bins=20, alpha=0.5, label='too_tilted', color='orange', density=True)
    ax.set_title(name, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q2_delivery_entry.png", dpi=100); plt.close()
print(f"\n  Saved: {OUT_DIR}/q2_delivery_entry.png")

# ----------------------------------------------------------------
# Q2.b: Predictive power — separability test
# ----------------------------------------------------------------
print(f"\n  Separability test (mean diff / pooled std → effect size):")
print(f"  {'Feature':25s}  {'success vs box_dropped':>24s}  {'success vs too_tilted':>24s}")
for name, fi in feat_names:
    s_v = delivery_entry[mask_success & mask_reached_delivery, fi]
    b_v = delivery_entry[mask_box_dropped & mask_reached_delivery, fi]
    t_v = delivery_entry[mask_too_tilted & mask_reached_delivery, fi]
    def cohens_d(a, b):
        if len(a) < 2 or len(b) < 2: return float('nan')
        pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
        if pooled < 1e-6: return 0.0
        return (a.mean() - b.mean()) / pooled
    d_sb = cohens_d(s_v, b_v); d_st = cohens_d(s_v, t_v)
    print(f"  {name:25s}  d={d_sb:>+6.3f}            d={d_st:>+6.3f}")

print(f"\n  |d| > 0.5 = medium effect, > 0.8 = large effect")
print(f"\n  Interpretation:")
print(f"  - If |d| LARGE on tilt/ang_vel for success vs failures → CLIMB-inherited instability is the cause")
print(f"  - If |d| SMALL (entry indistinguishable) → DELIVERY policy itself is the cause (not climb propagation)")

# ----------------------------------------------------------------
# Save summary
# ----------------------------------------------------------------
with open(f"{OUT_DIR}/summary.txt", "w") as f:
    f.write(f"Total missions: {N}\n")
    f.write(f"Outcomes: {dict(zip(*np.unique(outcomes, return_counts=True)))}\n\n")
    f.write(f"Q1: CLIMB failures: {n_climb_fail}/{N} ({100*n_climb_fail/N:.1f}%)\n")
    f.write(f"Q2: Of those reaching delivery ({n_reached_delivery}):\n")
    f.write(f"    success: {n_success} ({100*n_success/max(n_reached_delivery,1):.1f}%)\n")
    f.write(f"    box_dropped: {n_box_dropped} ({100*n_box_dropped/max(n_reached_delivery,1):.1f}%)\n")
    f.write(f"    too_tilted: {n_too_tilted} ({100*n_too_tilted/max(n_reached_delivery,1):.1f}%)\n")
print(f"\n  Saved: {OUT_DIR}/summary.txt")
print(f"\nDone.")
