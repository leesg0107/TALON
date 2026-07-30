"""Can a state-CONDITIONED close (PID condition) work?

Key question: do shallow-destined missions ever VISIT an instantaneous pre-close state
from which deep is predicted? If never, a close-condition never fires for exactly the
missions that need it -> degenerates to abort/retry.

Per-STEP instantaneous classifier (7-D state), GroupKFold by mission (no leakage),
then per-mission opportunity statistics.
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

d = np.load("logs/diagnose_climb_propagation_raw.npz", allow_pickle=True)
traces = d["dock_traces"]
target = d["climb_entry"][:, 15]

DEEP_T, SHAL_T = -0.020, -0.015
rows, labels, groups, mission_label = [], [], [], {}
for i, tr in enumerate(traces):
    if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5 or np.isnan(target[i]):
        continue
    y = target[i]
    if y < DEEP_T:
        lab = 1
    elif y > SHAL_T:
        lab = 0
    else:
        continue  # drop mid band, same as diagnosis convention
    tr = np.asarray(tr, dtype=np.float64)
    hit = np.where(tr[:, 0] >= 225)[0]
    pre = tr[:hit[0]] if len(hit) else tr
    pre = pre[(pre[:, 0] >= 150) & (pre[:, 0] < 225)]
    if len(pre) < 3:
        continue
    for row in pre:
        rows.append(row[1:8])  # drone_z, box_z, tilt, off_x, off_y, off_z, dist
        labels.append(lab)
        groups.append(i)
    mission_label[i] = lab

X = np.array(rows); yb = np.array(labels); g = np.array(groups)
print(f"steps={len(X)}  missions={len(mission_label)}  "
      f"deep={sum(v==1 for v in mission_label.values())}  shallow={sum(v==0 for v in mission_label.values())}")

# out-of-fold per-step P(deep), grouped by mission
oof = np.zeros(len(X))
gkf = GroupKFold(n_splits=5)
for tr_idx, te_idx in gkf.split(X, yb, g):
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X[tr_idx], yb[tr_idx])
    oof[te_idx] = clf.predict_proba(X[te_idx])[:, 1]

print(f"per-STEP instantaneous AUC (mission-grouped OOF): {roc_auc_score(yb, oof):.3f}")

# mission-level aggregates
mids = np.array(sorted(mission_label))
mmax = np.array([oof[g == i].max() for i in mids])
mmean = np.array([oof[g == i].mean() for i in mids])
mlab = np.array([mission_label[i] for i in mids])
print(f"mission-level AUC from max-prob: {roc_auc_score(mlab, mmax):.3f}, mean-prob: {roc_auc_score(mlab, mmean):.3f}")

# choose tau so that closing at a step with P>tau is 'trustworthy':
# precision of deep among steps above tau
for tau in (0.5, 0.7, 0.8, 0.9):
    above = oof > tau
    prec = yb[above].mean() if above.any() else float('nan')
    # opportunity: fraction of missions in each class with >=1 and >=5 steps above tau
    opp1 = {c: np.mean([np.sum((g == i) & above) >= 1 for i in mids[mlab == c]]) for c in (0, 1)}
    opp5 = {c: np.mean([np.sum((g == i) & above) >= 5 for i in mids[mlab == c]]) for c in (0, 1)}
    print(f"tau={tau}: step-precision(deep)={prec:.2f} | missions w/ >=1 step above: "
          f"deep {opp1[1]*100:.0f}% vs shallow {opp1[0]*100:.0f}% | >=5 steps: "
          f"deep {opp5[1]*100:.0f}% vs shallow {opp5[0]*100:.0f}%")

# trait confound check: DR mass vs depth
dr = d["dr_samples"]
mask = ~np.isnan(target)
for j, nm in enumerate(["mass_scale", "payload"]):
    print(f"corr({nm}, final depth) = {np.corrcoef(dr[mask, j], target[mask])[0,1]:+.3f}")

# within-mission variation: does P(deep) actually move within a mission's hold?
spread = np.array([oof[g == i].max() - oof[g == i].min() for i in mids])
print(f"within-mission P(deep) spread: median={np.median(spread):.2f}, "
      f"shallow-missions median={np.median(spread[mlab==0]):.2f}, deep={np.median(spread[mlab==1]):.2f}")
