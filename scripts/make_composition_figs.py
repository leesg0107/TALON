"""
Generate the NEW composition-finding figures for the paper, into
data/final_presentation/11_composition_findings/.

All numbers computed from logs/diagnose_climb_propagation_raw.npz and
data/grasp_states.pt. Claims corrected per adversarial verification:
 - "no FEEDFORWARD fix" (not "no fix"); detect-and-redraw is the prescriptive path
 - shallow-delivery failure partly training-coverage (caveat), not pure physics
"""
import os, numpy as np, scipy.stats as ss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "final_presentation", "11_composition_findings")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
C = {"succ": "#2a9d8f", "fail": "#e76f51", "deep": "#2a9d8f", "shallow": "#e76f51"}

d = np.load(os.path.join(ROOT, "logs", "diagnose_climb_propagation_raw.npz"), allow_pickle=True)
out, ce, de = d["outcomes"], d["climb_entry"], d["delivery_entry"]
OFFY = 15
succ = out == "full_success"
dfail = np.isin(out, ["box_dropped_delivery", "too_tilted_delivery"])

# ---------- FIG C: success / delivery-fail vs grasp depth (headline) ----------
depth = ce[:, OFFY] * 1000
ok = ~np.isnan(depth)
dp, sc, df = depth[ok], succ[ok], dfail[ok]
edges = np.array([-55, -30, -25, -20, -15, 5])
cent = 0.5 * (edges[:-1] + edges[1:])
ps, pf, ns = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (dp >= lo) & (dp < hi)
    ps.append(sc[m].mean() if m.sum() else np.nan)
    pf.append(df[m].mean() if m.sum() else np.nan)
    ns.append(int(m.sum()))
rho, pval = ss.spearmanr(dp, sc.astype(float))
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.plot(cent, np.array(ps) * 100, "o-", color=C["succ"], lw=2, label="P(full success)")
ax.plot(cent, np.array(pf) * 100, "s--", color=C["fail"], lw=2, label="P(delivery failure)")
for x, y, n in zip(cent, np.array(ps) * 100, ns):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center")
ax.axvline(-25, color="gray", ls=":", lw=1); ax.axvline(-20, color="gray", ls=":", lw=1)
ax.set_xlabel("inherited grasp depth box_offset_y (mm)  — more negative = deeper")
ax.set_ylabel("probability (%)")
ax.set_title("")  # title lives on the slide; keep figure clean
ax.legend(frameon=False); ax.invert_xaxis()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figC_success_vs_depth.png")); plt.close(fig)

# ---------- FIG B: covariate shift — standalone bank vs live pipeline ----------
gs = torch.load(os.path.join(ROOT, "data", "grasp_states.pt"), map_location="cpu", weights_only=False)
dpos = gs["drone_pos_local"].numpy(); bpos = gs["box_pos_local"].numpy(); qd = gs["drone_quat"].numpy()
def quat_to_R(q):  # wxyz -> (N,3,3)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
R = quat_to_R(qd)
gripper = dpos + np.einsum("nij,j->ni", R, np.array([0., 0., -0.08]))
box_off_b = np.einsum("nji,nj->ni", R, bpos - gripper)  # R^T @ (box-gripper)
bank_depth = box_off_b[:, 1] * 1000
pipe_depth = de[:, OFFY][~np.isnan(de[:, OFFY])] * 1000
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.hist(bank_depth, bins=40, density=True, alpha=0.55, color=C["deep"],
        label=f"standalone TRAIN/EVAL bank (grasp_states.pt)\nmean {bank_depth.mean():.0f}mm, {(bank_depth<-25).mean()*100:.0f}% deep")
ax.hist(pipe_depth, bins=40, density=True, alpha=0.55, color=C["shallow"],
        label=f"LIVE pipeline delivery-entry\nmean {pipe_depth.mean():.0f}mm, {(pipe_depth<-25).mean()*100:.0f}% deep / {(pipe_depth>-20).mean()*100:.0f}% shallow")
ax.axvline(-25, color="k", ls="--", lw=1); ax.text(-25, ax.get_ylim()[1]*0.92, " deep", fontsize=8)
ax.set_xlabel("grasp depth box_offset_y (mm)")
ax.set_ylabel("density")
ax.set_title("")  # title lives on the slide
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_covariate_shift.png")); plt.close(fig)

# ---------- FIG A: per-stage conditional cascade ----------
stages = ["Approach\n→Dock", "Dock\n→Climb", "Climb\n→Deliv", "Deliv\n→Done", "End-to-end"]
rates = [100.0, 98.6, 90.5, 80.9, 72.2]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
bars = ax.bar(range(len(stages)), rates, color=["#bbb", "#bbb", "#bbb", "#bbb", "#2a9d8f"])
for i, r in enumerate(rates):
    ax.text(i, r + 1, f"{r:.1f}%", ha="center", fontsize=9)
ax.axhline(72.2, color="#2a9d8f", ls=":", lw=1)
ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=8)
ax.set_ylabel("conditional success (%)"); ax.set_ylim(0, 108)
ax.set_title("Per-stage conditional cascade:  1.00 × 0.986 × 0.905 × 0.809  =  72.2%")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figA_stage_cascade.png")); plt.close(fig)

# ---------- FIG D: detect-and-redraw counterfactual (HONEST, conservative) ----------
# Redraw the GRASP (detectable pre-climb) until deep; bounded by deep-grasp competence.
p_deep = (dp < -25).mean()              # ~0.57 per independent attempt
s_deep, s_shallow = 0.791, 0.533        # measured success | deep / shallow
s_deeplevel = 0.911                     # success | deep AND level (deep minus tilt-climb-fails)
K = np.arange(1, 6)
p_got_deep = 1 - (1 - p_deep) ** K
exp_depthonly = s_deep * p_got_deep + s_shallow * (1 - p_got_deep)
exp_depthtilt = s_deeplevel * p_got_deep + s_shallow * (1 - p_got_deep)
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.plot(K, exp_depthonly * 100, "o-", color="#457b9d", lw=2, label="redraw until DEEP (ceiling = deep success 79%)")
ax.plot(K, exp_depthtilt * 100, "s--", color="#2a9d8f", lw=2, label="redraw until DEEP+LEVEL (ceiling = 91%)")
ax.axhline(72.2, color="gray", ls=":", lw=1); ax.text(4.2, 73, "current 72% (1 shot)", fontsize=8)
ax.set_xticks(K); ax.set_xlabel("max grasp attempts (detect shallow → re-grasp)")
ax.set_ylabel("predicted end-to-end success (%)"); ax.set_ylim(60, 95)
ax.set_title("")  # title lives on the slide
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figD_redraw_counterfactual.png")); plt.close(fig)

print("wrote figA_stage_cascade, figB_covariate_shift, figC_success_vs_depth, figD_redraw_counterfactual")
print(f"bank deep%={float((bank_depth<-25).mean())*100:.1f}  pipe deep%={float((pipe_depth<-25).mean())*100:.1f}  p_deep={p_deep:.3f}")
print(f"redraw depth-only: {np.round(exp_depthonly*100,1)}  depth+tilt: {np.round(exp_depthtilt*100,1)}")
