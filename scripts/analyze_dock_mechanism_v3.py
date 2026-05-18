"""V3 analyzer — uncover the TRUE physical mechanism of dock bifurcation.

Loads diagnose_dock_mechanism_raw.npz (14-D per-step dock features).
Tests hypotheses about WHAT causes box_offset_y to diverge at contain 230-280.

Features (14-D):
  0: contain_hold
  1: drone_z_local
  2: box_z_local
  3: yaw (rad)
  4: plate_l_angle
  5: plate_r_angle
  6: box_offset_x (body, strut axis)
  7: box_offset_y (body, plate axis) — KEY OUTCOME
  8: box_offset_z (body, vertical)
  9: drone-box x (world)
  10: drone-box y (world)
  11: box_vel_x (world)
  12: box_vel_y (world)
  13: drone_tilt_deg

Test hypotheses for what determines final box_offset_y at contain=325:
  H1: Drone YAW determines body-Y direction → final box_offset_y sign correlated with yaw
  H2: Plate close ASYMMETRY (plate_l vs plate_r diff) pushes box toward faster-closing plate
  H3: Drone xy position relative to box (in WORLD) determines which side box wedges
  H4: Box velocity early in dock predicts wedging direction
  H5: DR sample (mass_scale) creates altitude variance → different plate-box contact
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "logs/diagnose_dock_mechanism_raw.npz"
OUT = "logs/dock_mechanism_v3"
os.makedirs(OUT, exist_ok=True)

F_CONT, F_DZ, F_BZ, F_YAW, F_PL, F_PR, F_OFFX, F_OFFY, F_OFFZ, F_DBX, F_DBY, F_BVX, F_BVY, F_TILT = range(14)

d = np.load(DATA, allow_pickle=True)
outcomes = d["outcomes"]
dock_traces = d["dock_traces"]
final_y = d["final_box_offset_y"]
dr_samples = d["dr_samples"]

N = len(outcomes)
print(f"Loaded {N} missions\n")

# Filter to missions with valid dock trace + final_y
has_trace = np.array([tr is not None and len(tr) > 50 for tr in dock_traces])
has_final = ~np.isnan(final_y)
valid = has_trace & has_final
print(f"Valid for analysis: {valid.sum()}")

idx_valid = np.where(valid)[0]

# Group by FINAL box_offset_y (regardless of mission outcome — pure DOCK mechanism)
deep_mask = valid & (final_y < -0.020)
shallow_mask = valid & (final_y > -0.015)
print(f"  DEEP (final_y < -0.020): {deep_mask.sum()}")
print(f"  SHALLOW (final_y > -0.015): {shallow_mask.sum()}\n")

# Helper: get feature value at specific contain_hold
def get_at_contain(i, target_c, fi, tolerance=5):
    tr = dock_traces[i]
    mask = np.abs(tr[:, F_CONT] - target_c) <= tolerance
    if mask.any():
        return tr[mask, fi].mean()
    return float('nan')

# Helper: aggregate over contain window
def get_window(i, c_lo, c_hi, fi):
    tr = dock_traces[i]
    mask = (tr[:, F_CONT] >= c_lo) & (tr[:, F_CONT] <= c_hi)
    if mask.any():
        return tr[mask, fi].mean()
    return float('nan')

# Cohen's d
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2: return float('nan')
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    if pooled < 1e-6: return 0.0
    return (a.mean() - b.mean()) / pooled

# ----------------------------------------------------------------
# H1-H5: What feature at PRE-BIFURCATION window (contain 150-220)
# differs between deep and shallow?
# ----------------------------------------------------------------
print("=" * 70)
print("  Feature comparison: PRE-bifurcation (contain 150-220)")
print("=" * 70)

deep_idx = np.where(deep_mask)[0]
shallow_idx = np.where(shallow_mask)[0]

features_to_test = [
    ("yaw (rad)", F_YAW),
    ("plate_l_angle", F_PL),
    ("plate_r_angle", F_PR),
    ("|plate_l - plate_r|", "plate_diff"),
    ("box_offset_x (strut)", F_OFFX),
    ("box_offset_y (plate)", F_OFFY),
    ("box_offset_z (vertical)", F_OFFZ),
    ("drone-box x (world)", F_DBX),
    ("drone-box y (world)", F_DBY),
    ("box_vel_x (world)", F_BVX),
    ("box_vel_y (world)", F_BVY),
    ("drone_tilt_deg", F_TILT),
    ("drone_z_local", F_DZ),
]

print(f"  {'Feature':28s}  {'DEEP':>22s}  {'SHALLOW':>22s}  {'Cohen d':>8s}")
for name, fi in features_to_test:
    deep_vals = []
    for i in deep_idx:
        tr = dock_traces[i]
        mask = (tr[:, F_CONT] >= 150) & (tr[:, F_CONT] <= 220)
        if mask.any():
            if fi == "plate_diff":
                deep_vals.append(np.abs(tr[mask, F_PL] - tr[mask, F_PR]).mean())
            else:
                deep_vals.append(tr[mask, fi].mean())
    shallow_vals = []
    for i in shallow_idx:
        tr = dock_traces[i]
        mask = (tr[:, F_CONT] >= 150) & (tr[:, F_CONT] <= 220)
        if mask.any():
            if fi == "plate_diff":
                shallow_vals.append(np.abs(tr[mask, F_PL] - tr[mask, F_PR]).mean())
            else:
                shallow_vals.append(tr[mask, fi].mean())
    deep_vals, shallow_vals = np.array(deep_vals), np.array(shallow_vals)
    if len(deep_vals) > 1 and len(shallow_vals) > 1:
        d_eff = cohens_d(deep_vals, shallow_vals)
        print(f"  {name:28s}  μ={deep_vals.mean():+.4f}±{deep_vals.std():.4f}    "
              f"μ={shallow_vals.mean():+.4f}±{shallow_vals.std():.4f}    d={d_eff:+.3f}")

# ----------------------------------------------------------------
# H_special: Is final_box_offset_y SIGN correlated with drone yaw or
# drone-box world direction?
# ----------------------------------------------------------------
print()
print("=" * 70)
print("  Final box_offset_y SIGN vs drone state at contain=200")
print("=" * 70)

# For valid missions, get final_y and compare to drone state
yaw_at_200 = np.array([get_at_contain(i, 200, F_YAW) for i in idx_valid])
dbx_at_200 = np.array([get_at_contain(i, 200, F_DBX) for i in idx_valid])
dby_at_200 = np.array([get_at_contain(i, 200, F_DBY) for i in idx_valid])
fy_at_300  = np.array([get_at_contain(i, 300, F_OFFY) for i in idx_valid])  # final-ish

valid_arr = ~np.isnan(yaw_at_200) & ~np.isnan(fy_at_300)
yaw = yaw_at_200[valid_arr]
fy = fy_at_300[valid_arr]
dbx = dbx_at_200[valid_arr]
dby = dby_at_200[valid_arr]

# Correlate
corr_yaw = np.corrcoef(yaw, fy)[0, 1] if len(yaw) > 5 else float('nan')
corr_dbx = np.corrcoef(dbx, fy)[0, 1] if len(dbx) > 5 else float('nan')
corr_dby = np.corrcoef(dby, fy)[0, 1] if len(dby) > 5 else float('nan')
print(f"  corr(yaw @200, final_y):           {corr_yaw:+.3f}")
print(f"  corr(drone-box_x @200, final_y):   {corr_dbx:+.3f}")
print(f"  corr(drone-box_y @200, final_y):   {corr_dby:+.3f}")

# Check yaw vs (drone-box direction projection on body Y)
# Body Y unit vector in world: (-sin(yaw), cos(yaw))
# drone-box in body Y: -sin(yaw)*dbx + cos(yaw)*dby
db_in_bodyY = -np.sin(yaw) * dbx + np.cos(yaw) * dby
corr_dby_body = np.corrcoef(db_in_bodyY, fy)[0, 1] if len(db_in_bodyY) > 5 else float('nan')
print(f"  corr((drone-box) projected onto body-Y, final_y): {corr_dby_body:+.3f}")
print(f"   ↑ This should be HIGH if box gets pushed away from drone's body-Y offset direction")

# ----------------------------------------------------------------
# DR correlation
# ----------------------------------------------------------------
print()
print("  DR sample (mass_scale, payload_mass) vs final_y:")
mass_scale = dr_samples[idx_valid][valid_arr, 0]
payload = dr_samples[idx_valid][valid_arr, 1]
corr_mass = np.corrcoef(mass_scale, fy)[0, 1] if len(mass_scale) > 5 else float('nan')
corr_pl = np.corrcoef(payload, fy)[0, 1] if len(payload) > 5 else float('nan')
print(f"  corr(mass_scale, final_y):    {corr_mass:+.3f}")
print(f"  corr(payload_mass, final_y):  {corr_pl:+.3f}")

# ----------------------------------------------------------------
# Plot — feature trajectories
# ----------------------------------------------------------------
def bin_by_contain(idx_arr, fi, c_range=range(50, 326, 4)):
    bins = {c: [] for c in c_range}
    for i in idx_arr:
        tr = dock_traces[i]
        for row in tr:
            c = int(row[F_CONT])
            if c in bins:
                if fi == "plate_diff":
                    bins[c].append(abs(row[F_PL] - row[F_PR]))
                else:
                    bins[c].append(row[fi])
    cs, mu, sd = [], [], []
    for c, vs in sorted(bins.items()):
        if len(vs) >= 3:
            cs.append(c); mu.append(np.mean(vs)); sd.append(np.std(vs))
    return np.array(cs), np.array(mu), np.array(sd)

plots = [
    ("box_offset_y (body)", F_OFFY),
    ("plate_l angle", F_PL),
    ("plate_r angle", F_PR),
    ("|plate_l - plate_r|", "plate_diff"),
    ("box_offset_x (body)", F_OFFX),
    ("drone-box y (world)", F_DBY),
    ("drone yaw", F_YAW),
    ("box_vel_y (world)", F_BVY),
    ("drone_z_local", F_DZ),
]
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for ax, (name, fi) in zip(axes, plots):
    for mask, label, color in [(deep_mask, "DEEP", "green"),
                                 (shallow_mask, "SHALLOW", "red")]:
        cs, mu, sd = bin_by_contain(np.where(mask)[0], fi)
        if len(cs) > 0:
            ax.plot(cs, mu, color=color, label=f"{label} (n={mask.sum()})", linewidth=2)
            ax.fill_between(cs, mu - sd, mu + sd, color=color, alpha=0.15)
    ax.axvline(150, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(230, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(280, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("contain_hold")
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/v3_trajectories.png", dpi=100)
plt.close()
print(f"\n  Saved: {OUT}/v3_trajectories.png")
print("\nDone.")
