"""Definitive test: how predictable is FINAL grasp depth from PRE-CLOSE state?

Close trigger = contain 225 (drone_env.py:215). Only trace rows with contain < 225 used.
Out-of-fold (5-fold CV) R2 / AUC, linear vs gradient boosting, both npz datasets.
"""
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, roc_auc_score, roc_curve

rng = np.random.RandomState(0)

def build_features(traces, feat_names, windows=((100, 150), (150, 200), (200, 225))):
    """Per-mission features from pre-close rows only (contain col 0 < 225)."""
    rows, ok = [], []
    for tr in traces:
        if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5:
            rows.append(None); ok.append(False); continue
        tr = np.asarray(tr, dtype=np.float64)
        pre = tr[tr[:, 0] < 225]
        if len(pre) < 5:
            rows.append(None); ok.append(False); continue
        f, names = [], []
        for lo, hi in windows:
            m = (pre[:, 0] >= lo) & (pre[:, 0] < hi)
            seg = pre[m]
            for j, nm in enumerate(feat_names):
                if j == 0:
                    continue
                if m.sum() >= 2:
                    f += [seg[:, j].mean(), seg[:, j].std()]
                else:
                    f += [np.nan, np.nan]
                names += [f"{nm}_w{lo}m", f"{nm}_w{lo}s"]
        last = pre[np.argsort(pre[:, 0])][-3:]  # last 3 pre-close rows
        for j, nm in enumerate(feat_names):
            if j == 0:
                continue
            f.append(last[:, j].mean()); names.append(f"{nm}_last")
        rows.append(np.array(f)); ok.append(True)
    return rows, np.array(ok), names

def evaluate(X, y, label, shallow_mask, deep25_mask):
    kf = KFold(5, shuffle=True, random_state=0)
    # impute NaN with column median (some missions lack early windows)
    med = np.nanmedian(X, axis=0)
    Xi = np.where(np.isnan(X), med, X)

    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    gbm = HistGradientBoostingRegressor(random_state=0)
    p_lin = cross_val_predict(ridge, Xi, y, cv=kf)
    p_gbm = cross_val_predict(gbm, Xi, y, cv=kf)
    print(f"\n=== {label} (N={len(y)}) ===")
    print(f"  continuous depth : ridge R2={r2_score(y, p_lin):+.3f}   GBM R2={r2_score(y, p_gbm):+.3f}")

    for cls_name, pos in [("SHALLOW(> -15mm) vs rest", shallow_mask),
                          ("NOT-DEEP(> -25mm) vs deep", deep25_mask)]:
        yb = pos.astype(int)
        if yb.sum() < 10 or (1 - yb).sum() < 10:
            print(f"  {cls_name}: skipped (class too small)")
            continue
        logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        gbc = HistGradientBoostingClassifier(random_state=0)
        s_lin = cross_val_predict(logit, Xi, yb, cv=kf, method="predict_proba")[:, 1]
        s_gbm = cross_val_predict(gbc, Xi, yb, cv=kf, method="predict_proba")[:, 1]
        auc_l, auc_g = roc_auc_score(yb, s_lin), roc_auc_score(yb, s_gbm)
        s_best = s_gbm if auc_g >= auc_l else s_lin
        fpr, tpr, _ = roc_curve(yb, s_best)
        tpr_at5 = np.interp(0.05, fpr, tpr)
        tpr_at10 = np.interp(0.10, fpr, tpr)
        print(f"  {cls_name}: logit AUC={auc_l:.3f}  GBM AUC={auc_g:.3f}  "
              f"| best model catches {tpr_at5*100:.0f}% @5%FPR, {tpr_at10*100:.0f}% @10%FPR "
              f"(base rate {yb.mean()*100:.0f}%)")
    return Xi

def single_feature_report(Xi, names, y, shallow_mask, topk=8):
    cors = []
    for j, nm in enumerate(names):
        c = np.corrcoef(Xi[:, j], y)[0, 1]
        a = roc_auc_score(shallow_mask.astype(int), Xi[:, j])
        a = max(a, 1 - a)
        cors.append((abs(c), c, a, nm))
    cors.sort(reverse=True)
    print("  top single features (|corr with depth|, AUC shallow):")
    for _, c, a, nm in cors[:topk]:
        print(f"    {nm:22s} corr={c:+.3f}  AUC={a:.3f}")

# ---------- Dataset 1: climb propagation npz, 8-D traces, N=400 ----------
d1 = np.load("logs/diagnose_climb_propagation_raw.npz", allow_pickle=True)
names8 = ["contain", "drone_z", "box_z", "tilt", "off_x", "off_y", "off_z", "dist"]
rows, ok, fnames = build_features(d1["dock_traces"], names8)
target = d1["climb_entry"][:, 15]
valid = ok & ~np.isnan(target)
X = np.vstack([r for r, o in zip(rows, valid) if o and r is not None])
y = target[valid]
# append DR samples
X = np.hstack([X, d1["dr_samples"][valid]])
fnames_full = fnames + ["mass_scale", "payload"]
shallow = y > -0.015
deep25 = y > -0.025
Xi = evaluate(X, y, "8-D traces (climb-propagation npz)", shallow, deep25)
single_feature_report(Xi, fnames_full, y, shallow)

# ---------- Dataset 2: v3 dock mechanism npz, 14-D traces, N=300 ----------
d2 = np.load("logs/diagnose_dock_mechanism_raw.npz", allow_pickle=True)
names14 = ["contain", "drone_z", "box_z", "yaw", "plate_l", "plate_r",
           "off_x", "off_y", "off_z", "dbx_w", "dby_w", "bvx", "bvy", "tilt"]
rows2, ok2, fnames2 = build_features(d2["dock_traces"], names14)
t2 = d2["final_box_offset_y"]
valid2 = ok2 & ~np.isnan(t2)
X2 = np.vstack([r for r, o in zip(rows2, valid2) if o and r is not None])
y2 = t2[valid2]
X2 = np.hstack([X2, d2["dr_samples"][valid2]])
fnames2_full = fnames2 + ["mass_scale", "payload"]
shallow2 = y2 > -0.015
deep252 = y2 > -0.025
Xi2 = evaluate(X2, y2, "14-D traces (dock-mechanism v3 npz)", shallow2, deep252)
single_feature_report(Xi2, fnames2_full, y2, shallow2)

# ---------- Robustness: temporal-prefix filter + monotonicity + permutation ----------
print("\n\n########## ROBUSTNESS CHECKS ##########")
tr_all = d1["dock_traces"]
nonmono = 0
for tr in tr_all:
    if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5:
        continue
    c = np.asarray(tr)[:, 0]
    if np.any(np.diff(c) < -5):
        nonmono += 1
print(f"traces with contain drop >5 (reset events): {nonmono}")

def build_prefix(traces, feat_names):
    """Rows strictly BEFORE first contain>=225 in temporal order."""
    rows, ok = [], []
    for tr in traces:
        if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5:
            rows.append(None); ok.append(False); continue
        tr = np.asarray(tr, dtype=np.float64)
        hit = np.where(tr[:, 0] >= 225)[0]
        pre = tr[:hit[0]] if len(hit) else tr
        if len(pre) < 5:
            rows.append(None); ok.append(False); continue
        f = []
        for lo, hi in ((100, 150), (150, 200), (200, 225)):
            m = (pre[:, 0] >= lo) & (pre[:, 0] < hi)
            seg = pre[m]
            for j in range(1, len(feat_names)):
                f += [seg[:, j].mean(), seg[:, j].std()] if m.sum() >= 2 else [np.nan, np.nan]
        f += list(pre[-3:, 1:].mean(axis=0))
        rows.append(np.array(f)); ok.append(True)
    return rows, np.array(ok)

rowsP, okP = build_prefix(tr_all, names8)
validP = okP & ~np.isnan(target)
XP = np.vstack([r for r, o in zip(rowsP, validP) if o and r is not None])
yP = target[validP]
XP = np.hstack([XP, d1["dr_samples"][validP]])
shallowP = yP > -0.015
evaluate(XP, yP, "8-D PREFIX-filtered (no reset leakage)", shallowP, yP > -0.025)

# permutation sanity: shuffle target
yshuf = yP.copy(); rng.shuffle(yshuf)
evaluate(XP, yshuf, "8-D PREFIX, SHUFFLED target (should be ~0 / 0.5)", yshuf > -0.015, yshuf > -0.025)
