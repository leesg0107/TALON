"""
Prescription-validation analysis (Phase A, retrospective, ZERO GPU).

Tests the thesis claim that a PHASE-TRANSITION ENTRY-STATE VALIDATION GATE
predictably separates the matched downstream failure class, using existing
diagnostic data only (no re-simulation).

Data: logs/diagnose_climb_propagation_raw.npz  (N=400 missions)
  climb_entry[:,18], delivery_entry[:,18]  snapshots (snapshot_state ordering):
    0-2 drone_pos_local, 3-5 vel_w, 6 tilt_deg, 7 ang_vel_norm,
    8-10 box_pos_local, 11-13 obj_vel, 14-16 box_offset_b(xyz), 17 drone_box_dist

Outputs: data/final_presentation/10_prescription_validation/
  fig1_composition_propagation.png   - grasp depth climb->delivery (headline)
  fig2_gate_roc.png                  - 4 class-specific gate ROC curves
  fig3_entry_distributions.png       - entry-state distributions success vs class
  fig4_specificity_matrix.png        - gate x class AUC (class-specificity / R4)
  fig5_intraphase_falsified.png      - end-to-end variance band + falsified arm
  stats.txt                          - full numeric report
"""
import os, sys
import numpy as np
import scipy.stats as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "logs", "diagnose_climb_propagation_raw.npz")
OUT = os.path.join(ROOT, "data", "final_presentation", "10_prescription_validation")
os.makedirs(OUT, exist_ok=True)

# feature column indices in the 18-D snapshot
TILT, ANGV, VELZ, OFFY = 6, 7, 5, 15
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- stats helpers
def clean(x):
    x = np.asarray(x, float)
    return x[~np.isnan(x)]

def cohens_d(a, b):
    a, b = clean(a), clean(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp

def auc(score_pos, score_neg):
    """P(score_pos > score_neg); 'positive' = the failure class."""
    p, n = clean(score_pos), clean(score_neg)
    if len(p) == 0 or len(n) == 0:
        return np.nan
    r = ss.rankdata(np.concatenate([p, n]))
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))

def auc_ci(score_pos, score_neg, reps=2000):
    p, n = clean(score_pos), clean(score_neg)
    if len(p) == 0 or len(n) == 0:
        return (np.nan, np.nan)
    vals = [auc(RNG.choice(p, len(p)), RNG.choice(n, len(n))) for _ in range(reps)]
    return tuple(np.percentile(vals, [2.5, 97.5]))

def roc_points(score_pos, score_neg):
    p, n = clean(score_pos), clean(score_neg)
    thr = np.unique(np.concatenate([p, n]))[::-1]
    tpr = [np.mean(p >= t) for t in thr]
    fpr = [np.mean(n >= t) for t in thr]
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]

def fisher_2x2(gated_fail, total_fail, gated_succ, total_succ):
    table = [[gated_fail, total_fail - gated_fail],
             [gated_succ, total_succ - gated_succ]]
    return ss.fisher_exact(table)[1]

# ---------------------------------------------------------------- load
d = np.load(NPZ, allow_pickle=True)
out, ce, de = d["outcomes"], d["climb_entry"], d["delivery_entry"]
succ = out == "full_success"
cf   = out == "climb_failed"
bd   = out == "box_dropped_delivery"
tt   = out == "too_tilted_delivery"

# signal definitions: (name, transition, feature array+col, fail mask, orient)
# orient = +1 if higher feature => more failure, -1 if lower => more failure
SIGNALS = [
    ("DOCK->CLIMB  tilt -> climb_failed",        ce, TILT, cf, +1, "deg"),
    ("DOCK->CLIMB  grasp_depth -> box_dropped",  ce, OFFY, bd, +1, "mm"),   # shallower(higher) = fail
    ("DOCK->CLIMB  grasp_depth -> too_tilted",   ce, OFFY, tt, +1, "mm"),
    ("CLIMB->DELIV vel_z -> box_dropped",        de, VELZ, bd, -1, "m/s"),  # lower vel_z = fail
    ("CLIMB->DELIV ang_vel -> too_tilted",       de, ANGV, tt, +1, "rad/s"),
]

lines = []
def log(s=""):
    print(s); lines.append(s)

log("=" * 78)
log("PRESCRIPTION VALIDATION  -  Phase A retrospective gate-as-classifier")
log("=" * 78)
u, c = np.unique(out, return_counts=True)
log("Outcome counts (N=%d):" % len(out))
for k, v in zip(u, c):
    log("   %-26s %d" % (k, v))

log("\n--- Per-signal class-specific separation (ROC AUC, bootstrap 95%% CI) ---")
sig_stats = []
for name, arr, col, mask, orient, unit in SIGNALS:
    s_fail = orient * arr[mask, col]
    s_succ = orient * arr[succ, col]
    A = auc(s_fail, s_succ); lo, hi = auc_ci(s_fail, s_succ)
    dd = cohens_d(arr[mask, col], arr[succ, col])
    scale = 1000 if unit == "mm" else 1
    log("  %-42s  AUC=%.3f [%.3f,%.3f]  d=%+.2f  (succ %.2f vs fail %.2f %s, n=%d)"
        % (name, A, lo, hi, dd, arr[succ, col].mean() * scale,
           arr[mask, col].mean() * scale, unit, mask.sum()))
    sig_stats.append((name, arr, col, mask, orient, unit, A, lo, hi, dd))

# composition: does climb change grasp depth?
v = ~np.isnan(ce[:, OFFY]) & ~np.isnan(de[:, OFFY])
corr = np.corrcoef(ce[v, OFFY], de[v, OFFY])[0, 1]
drift = (de[v, OFFY] - ce[v, OFFY]) * 1000
log("\n--- COMPOSITION proof: grasp depth climb-entry -> delivery-entry ---")
log("  corr = %.4f   mean|drift| = %.4f mm   => shallow grasp is ~100%% DOCK-origin"
    % (corr, np.abs(drift).mean()))

# Fisher headline: tilt gate at fixed 5% false-rejection budget
tilt_succ = ce[succ, TILT]; tilt_fail = ce[cf, TILT]
thr = np.nanpercentile(tilt_succ, 95)          # reject top 5% of successes
cap = np.mean(clean(tilt_fail) >= thr)         # fraction of climb_failed captured
frr = np.mean(clean(tilt_succ) >= thr)
p = fisher_2x2((clean(tilt_fail) >= thr).sum(), len(clean(tilt_fail)),
               (clean(tilt_succ) >= thr).sum(), len(clean(tilt_succ)))
log("\n--- Operating point (DOCK->CLIMB tilt gate, <=5%% false-reject budget) ---")
log("  threshold=%.1f deg  capture(climb_failed)=%.0f%%  false-reject(succ)=%.0f%%  Fisher p=%.2e"
    % (thr, cap * 100, frr * 100, p))

# ============================================================ FIGURES
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
C = {"succ": "#2a9d8f", "fail": "#e76f51", "bd": "#e76f51", "tt": "#f4a261", "cf": "#c1121f"}

# fig1 -- composition propagation (headline)
fig, ax = plt.subplots(figsize=(6.2, 5.6))
for m, lbl, col in [(succ, "full_success", C["succ"]),
                    (bd, "box_dropped", C["bd"]), (tt, "too_tilted", C["tt"])]:
    mm = m & v
    ax.scatter(ce[mm, OFFY] * 1000, de[mm, OFFY] * 1000, s=18, alpha=0.6,
               label=lbl, color=col, edgecolors="none")
lim = [ce[v, OFFY].min() * 1000 - 2, ce[v, OFFY].max() * 1000 + 2]
ax.plot(lim, lim, "k--", lw=1, label="y = x (climb is identity)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("grasp depth at CLIMB entry  box_offset_y (mm)")
ax.set_ylabel("grasp depth at DELIVERY entry  box_offset_y (mm)")
ax.set_title("")  # title lives on the slide
ax.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_composition_propagation.png")); plt.close(fig)

# fig2 -- gate ROC curves
fig, ax = plt.subplots(figsize=(6.2, 6.0))
for name, arr, col, mask, orient, unit, A, lo, hi, dd in sig_stats:
    fpr, tpr = roc_points(orient * arr[mask, col], orient * arr[succ, col])
    short = name.split("->")[-1].strip() if "->" in name else name
    short = name.split()[1] + " -> " + name.split("->")[-1].strip()
    ax.plot(fpr, tpr, lw=2, label="%s  (AUC %.2f [%.2f,%.2f])" % (short, A, lo, hi))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
ax.set_xlabel("false-rejection rate (successes wrongly gated)")
ax.set_ylabel("capture rate (failures gated)")
ax.set_title("Entry-state gate as class-specific classifier")
ax.legend(loc="lower right", frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_gate_roc.png")); plt.close(fig)

# fig3 -- entry-state distributions (4 panels)
panels = [
    ("CLIMB-entry tilt (deg)", ce, TILT, cf, "climb_failed", 1),
    ("CLIMB-entry grasp depth (mm)", ce, OFFY, bd | tt, "delivery fail", 1000),
    ("DELIVERY-entry vel_z (m/s)", de, VELZ, bd, "box_dropped", 1),
    ("DELIVERY-entry ang_vel (rad/s)", de, ANGV, tt, "too_tilted", 1),
]
fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.2))
for axp, (title, arr, col, mask, flbl, sc) in zip(axs.ravel(), panels):
    a = clean(arr[succ, col]) * sc; b = clean(arr[mask, col]) * sc
    bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 30)
    axp.hist(a, bins, alpha=0.6, color=C["succ"], density=True, label="success")
    axp.hist(b, bins, alpha=0.6, color=C["fail"], density=True, label=flbl)
    axp.axvline(a.mean(), color=C["succ"], ls="--", lw=1)
    axp.axvline(b.mean(), color=C["fail"], ls="--", lw=1)
    axp.set_title("%s  (d=%+.2f)" % (title, cohens_d(arr[mask, col], arr[succ, col])))
    axp.legend(frameon=False, fontsize=8); axp.set_yticks([])
fig.suptitle("Entry-state distributions: success vs matched failure class")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_entry_distributions.png")); plt.close(fig)

# fig4 -- class specificity matrix (gate x class AUC)
gates = [("tilt@climb", ce, TILT, +1), ("depth@climb", ce, OFFY, +1),
         ("vel_z@deliv", de, VELZ, -1), ("angv@deliv", de, ANGV, +1)]
classes = [("climb_failed", cf), ("box_dropped", bd), ("too_tilted", tt)]
M = np.full((len(gates), len(classes)), np.nan)
for i, (gn, arr, col, orient) in enumerate(gates):
    for j, (cn, m) in enumerate(classes):
        # delivery gates can't see climb_failed (no delivery entry); skip
        if "deliv" in gn and cn == "climb_failed":
            continue
        M[i, j] = auc(orient * arr[m, col], orient * arr[succ, col])
fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(M, vmin=0.3, vmax=0.9, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(classes))); ax.set_xticklabels([c[0] for c in classes], rotation=20)
ax.set_yticks(range(len(gates))); ax.set_yticklabels([g[0] for g in gates])
for i in range(len(gates)):
    for j in range(len(classes)):
        txt = "n/a" if np.isnan(M[i, j]) else "%.2f" % M[i, j]
        ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_title("")  # title lives on the slide
fig.colorbar(im, label="AUC (failure-predicting orientation)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_specificity_matrix.png")); plt.close(fig)

# fig5 -- end-to-end variance band + falsified intra-phase arm
# verified numbers (N=500 unless noted) from logs/ and 08_interventions_tested
baselines = {"baseline run A (README)": 72.2, "baseline run B (intervention)": 68.4,
             "baseline run C (re-run)": 66.0}
inter = {"INTER-phase: attitude gate (Fix1)": 74.4}
intra = {"INTRA: clip 1.0": 32.5, "INTRA: clip 2.0": 48.9,
         "INTRA: barrier v1 (retrain)": 41.0, "INTRA: barrier v2 (retrain)": 66.8}
bl = np.array(list(baselines.values()))
fig, ax = plt.subplots(figsize=(9.0, 5.2))
order = list(baselines) + list(inter) + list(intra)
vals = [baselines[k] for k in baselines] + [inter[k] for k in inter] + [intra[k] for k in intra]
cols = ["#999"] * len(baselines) + ["#2a9d8f"] * len(inter) + ["#e76f51"] * len(intra)
ax.bar(range(len(order)), vals, color=cols)
ax.axhspan(bl.min(), bl.max(), color="#bbbbbb", alpha=0.35,
           label="same-config variance band (%.0f-%.0f%%, unseeded)" % (bl.min(), bl.max()))
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("end-to-end success (%)")
ax.set_title("Why end-to-end %% can't be the DV, and the falsified INTRA-phase arm\n"
             "intra-phase action constraints fall in/below the noise band; inter-phase gate barely clears it")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_intraphase_falsified.png")); plt.close(fig)

with open(os.path.join(OUT, "stats.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
log("\nFigures + stats.txt written to: %s" % OUT)
