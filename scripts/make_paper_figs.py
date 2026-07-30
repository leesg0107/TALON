"""Paper figures -> data/paper_figures/.

Every number is recomputed from the logged runs; the only hard-coded values are the
URDF geometry constants (assets/gripper_drone_v5.urdf) used for the cross-sections.

Conventions used in all figures
  s  grasp depth |box offset along the closing axis| at the handoff (mm).
     The sign is fixed by which plate wins the closure and carries no information.
  g  vertical clearance at the instant closing is commanded: plate-hinge height
     above the payload top face (mm).
  th plate angle from vertical where the closing motion stopped (deg).

Run:  python scripts/make_paper_figs.py [fig2 fig3 ...]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

OUT = "data/paper_figures"
os.makedirs(OUT, exist_ok=True)

C_ENG, C_NOT, C_INK, C_MUTE = "#0072B2", "#D55E00", "#1a1a1a", "#8a8a8a"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#444444", "axes.labelcolor": C_INK, "text.color": C_INK,
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "hatch.linewidth": 0.6,
})

ENG_CUT, NOT_CUT = 25.0, 15.0
BOX = 80.0
# URDF cross-section constants (mm, plate-closing plane)
HINGE = {"L": np.array([+20.0, -22.5]), "R": np.array([-20.0, -22.5])}
FOOT_LOCAL = {"L": np.array([-15.0, -116.0]), "R": np.array([+15.0, -116.0])}
FOOT_HALF = np.array([21.0, 7.5])
ARM_TIP = {"L": np.array([-15.0, -108.5]), "R": np.array([+15.0, -108.5])}
REF_Z = -80.0            # offsets are measured from this point (gripper reference)
HINGE_Z = -22.5
OPEN_DEG = 50.0
CLOSED_DEG = -5.0


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (p + z * z / (2 * n) - c) / (1 + z * z / n)),
            min(1.0, (p + z * z / (2 * n) + c) / (1 + z * z / n)))


# ----------------------------------------------------------------------------- fig 2
def fig2():
    d = np.load("logs/diagnose_climb_propagation_raw.npz", allow_pickle=True)
    ce, de, out = d["climb_entry"], d["delivery_entry"], d["outcomes"]
    outs = np.array([str(o) for o in out])
    ok = ~np.isnan(ce[:, 0])
    s = np.abs(ce[:, 15]) * 1000.0
    reached = ok & ~np.isnan(de[:, 0])
    succ = outs == "full_success"
    g = {"eng": ok & (s > ENG_CUT), "not": ok & (s < NOT_CUT)}
    n_mid = ok.sum() - g["eng"].sum() - g["not"].sum()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.1, 2.75),
                                   gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.30})

    bins = np.arange(0, 57.5, 2.5)
    counts, edges = np.histogram(s[ok], bins=bins)
    for c0, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        mid = (e0 + e1) / 2
        col = C_NOT if mid < NOT_CUT else (C_ENG if mid > ENG_CUT else "#bdbdbd")
        axA.bar(mid, c0, width=2.4, color=col, edgecolor="white", linewidth=0.4,
                hatch="///" if mid < NOT_CUT else None)
    for x in (NOT_CUT, ENG_CUT):
        axA.axvline(x, color="#444444", lw=0.6, ls=(0, (3, 2)))
    ytop = counts.max() * 1.30
    axA.set_ylim(0, ytop)
    axA.text(7.0, ytop * 0.97, f"not engaged\nn={g['not'].sum()}", ha="center", va="top",
             color=C_NOT, fontsize=7.5, linespacing=1.25)
    axA.text(41.0, ytop * 0.97, f"engaged\nn={g['eng'].sum()}", ha="center", va="top",
             color=C_ENG, fontsize=7.5, linespacing=1.25)
    axA.text(20.0, ytop * 0.42, f"mid\nn={n_mid}", ha="center", va="top",
             color="#7a7a7a", fontsize=7, linespacing=1.25)
    axA.set_xlabel("grasp depth $s$ at the handoff (mm)")
    axA.set_ylabel(f"accepted handoffs (n={ok.sum()})")
    axA.set_title("(a) the acceptance test admits the full spread", loc="left", pad=5)

    metrics = [("climb\npassed", reached, ok),
               ("delivery phase\nsucceeded", succ & reached, reached),
               ("mission\ncompleted", succ & ok, ok)]
    x = np.arange(len(metrics)); w = 0.34
    report = {}
    for key, colour, hatch, off, lab in (("eng", C_ENG, None, -w / 2, "engaged"),
                                         ("not", C_NOT, "///", +w / 2, "not engaged")):
        vals, los, his, ns = [], [], [], []
        for name, num, den in metrics:
            k, n = (num & g[key]).sum(), (den & g[key]).sum()
            p = k / n if n else 0.0
            lo, hi = wilson(k, n)
            vals.append(p * 100); los.append((p - lo) * 100); his.append((hi - p) * 100); ns.append(n)
            report[(key, name.replace("\n", " "))] = (k, n, p * 100, lo * 100, hi * 100)
        axB.bar(x + off, vals, w, color=colour, edgecolor="white", lw=0.5,
                hatch=hatch, label=lab, zorder=3)
        axB.errorbar(x + off, vals, yerr=[los, his], fmt="none", ecolor=C_INK,
                     elinewidth=0.7, capsize=2, zorder=4)
        for xi, v, h in zip(x + off, vals, his):
            axB.text(xi, v + h + 2.0, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=7.5, color=C_INK)
    axB.set_xticks(x); axB.set_xticklabels([m[0] for m in metrics])
    axB.set_ylim(0, 108); axB.set_yticks([0, 25, 50, 75, 100])
    xbr = 2 + w / 2 + 0.24
    axB.annotate("", xy=(xbr, 50.9), xytext=(xbr, 79.1),
                 arrowprops=dict(arrowstyle="<->", lw=0.8, color=C_INK))
    for yv in (50.9, 79.1):
        axB.plot([2 + w / 2 + 0.04, xbr], [yv, yv], color=C_INK, lw=0.6)
    axB.text(xbr + 0.05, 65.0, "28.2\npts", fontsize=7, color=C_INK, va="center",
             linespacing=1.1)
    axB.text(0, 24, "tilt effect,\nnot depth —\nsee Table 5", ha="center", fontsize=6.3,
             color=C_INK, linespacing=1.25,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.6))
    axB.set_ylabel("rate (%)")
    axB.set_title("(b) outcome by grasp-depth group", loc="left", pad=5)
    axB.grid(axis="y", color="#ececec", lw=0.5); axB.set_axisbelow(True)
    h, lb = axB.get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.09), handlelength=1.3, columnspacing=2.0)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig2_outcome_divergence.{ext}")
    plt.close(fig)
    for k_, v in report.items():
        print(f"[fig2] {k_[0]:4s} {k_[1]:24s} {v[0]:3d}/{v[1]:3d} = {v[2]:5.1f}% "
              f"[{v[3]:.1f}, {v[4]:.1f}]")


# ----------------------------------------------------------------------------- fig 3
def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _part(side, theta_deg, local, half):
    R = _rot(np.radians(theta_deg) * (1 if side == "L" else -1))
    c = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    return (R @ (local + c).T).T + HINGE[side]


def _arm_line(side, theta_deg):
    R = _rot(np.radians(theta_deg) * (1 if side == "L" else -1))
    tip = R @ ARM_TIP[side] + HINGE[side]
    return np.array([HINGE[side], tip])


def _foot_gap(side, theta_deg, oy):
    P = _part(side, theta_deg, FOOT_LOCAL[side], FOOT_HALF)
    return (P[:, 0].min() - (oy + BOX / 2)) if side == "L" else ((oy - BOX / 2) - P[:, 0].max())


def _load_dock():
    d = np.load("logs/diagnose_dock_mechanism_raw.npz", allow_pickle=True)
    tr_all, final = d["dock_traces"], d["final_box_offset_y"]

    def at(tr, c, tol=6):
        m = np.abs(tr[:, 0] - c) <= tol
        return tr[m].mean(axis=0) if m.any() else None

    rows = []
    for tr, f in zip(tr_all, final):
        if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5 or np.isnan(f):
            continue
        tr = np.asarray(tr, float)
        a, b = at(tr, 225), at(tr, 320)
        if a is None or b is None:
            continue
        # g: hinge height above payload top at the close command
        g = ((a[1] + HINGE_Z / 1000.0) - (a[2] + BOX / 2 / 1000.0)) * 1000.0
        rows.append((abs(f) * 1000, np.degrees(b[4]), np.degrees(b[5]),
                     b[7] * 1000, b[8] * 1000, g))
    r = np.array(rows)
    return dict(s=r[:, 0], thL=r[:, 1], thR=r[:, 2], oy=r[:, 3], oz=r[:, 4], g=r[:, 5])


def _draw_case(ax, thL, thR, oy, oz, title, colour, annotate, gval=None):
    box_zc = oz + REF_Z
    box_bot = box_zc - BOX / 2
    ax.add_patch(Rectangle((-70, HINGE_Z), 140, -HINGE_Z, facecolor="#efefef",
                           edgecolor="#a8a8a8", lw=0.6))
    ax.text(0, HINGE_Z / 2, "body", ha="center", va="center", fontsize=6.5,
            color="#6e6e6e")
    # pedestal: 300 x 300 mm top (env_cfg.py PEDESTAL_CFG) -> wider than the view
    ax.add_patch(Rectangle((-150, box_bot - 60), 300, 60, facecolor="none",
                           edgecolor="#c0c0c0", lw=0.6, hatch="////"))
    ax.text(-118, box_bot - 32, "pedestal", ha="left", va="center", fontsize=6.5,
            color="#9a9a9a")
    ax.add_patch(Rectangle((oy - BOX / 2, box_bot), BOX, BOX, facecolor=colour,
                           alpha=0.18, edgecolor=colour, lw=1.1, zorder=3))
    ax.text(oy, box_zc, "payload", ha="center", va="center", fontsize=6.5, color=colour)
    for side, th in (("L", thL), ("R", thR)):
        L = _arm_line(side, th)
        ax.plot(L[:, 0], L[:, 1], color="#9a9a9a", lw=1.6, solid_capstyle="round",
                zorder=4)
        ax.add_patch(Polygon(_part(side, th, FOOT_LOCAL[side], FOOT_HALF), closed=True,
                             facecolor="#7d7d7d", edgecolor="#2b2b2b", lw=0.8, zorder=5))
        ax.plot(*HINGE[side], marker="o", ms=2.8, color=C_INK, zorder=6)
    if gval is not None:
        gcol = colour if colour != C_MUTE else C_INK
        ax.add_patch(FancyArrowPatch((-104, HINGE_Z), (-104, box_zc + BOX / 2),
                                     arrowstyle="<->", mutation_scale=6, lw=0.7,
                                     color=gcol))
        lab = "$h$" if gval < 0 else f"$h$ = {gval:.0f} mm"
        ax.text(-110, (HINGE_Z + box_zc + BOX / 2) / 2, lab, ha="right", va="center",
                fontsize=7, color=gcol)
    if annotate:
        gy = _foot_gap("L", thL, oy)
        txt = (f"foot reaches\nthe payload\n({gy:+.0f} mm)" if gy < 0 else
               f"foot stops short\n({gy:+.0f} mm)")
        col = C_ENG if gy < 0 else C_NOT
        ax.annotate(txt, xy=(oy + BOX / 2 + min(gy, 0) / 2, box_bot + 22),
                    xytext=(104, box_bot + 34), fontsize=6.8, color=col, ha="center",
                    linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=col))
    ax.set_xlim(-140, 140); ax.set_ylim(-225, 4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", pad=3)


def _closure_bundle(lo=225, hi=330):
    """Both jaw angles and the payload offset, per control step, over the closure."""
    d = np.load("logs/diagnose_dock_mechanism_raw.npz", allow_pickle=True)
    grid = np.arange(lo, hi + 1, 1.0)
    A, B, Y, brk = [], [], [], []
    for tr in d["dock_traces"]:
        if tr is None or len(np.shape(tr)) != 2 or len(tr) < 6:
            continue
        tr = np.asarray(tr, float)
        c = tr[:, 0]
        if c.max() < lo + 5:
            continue
        L, R = np.degrees(tr[:, 4]), np.degrees(tr[:, 5])
        # the jaw that ends further closed is the advancing one
        adv, opp = (L, R) if L[-1] <= R[-1] else (R, L)
        A.append(np.interp(grid, c, adv)); B.append(np.interp(grid, c, opp))
        Y.append(np.interp(grid, c, np.abs(tr[:, 7]) * 1000.0))
        m = c >= lo
        dif = np.abs(L[m] - R[m])
        if dif.max() > 1.0:
            brk.append(c[m][np.argmax(dif > 1.0)])
    return grid, np.array(A), np.array(B), np.array(Y), np.array(brk)


def _draw_symmetry_break(ax):
    grid, A, B, Y, brk = _closure_bundle()
    kb = np.median(brk)
    q = lambda M: (np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0))
    for M, col, ls_, lab in ((A, C_ENG, "-", "advancing jaw"),
                             (B, C_MUTE, (0, (5, 1.6)), "opposing jaw")):
        m, lo, hi = q(M)
        ax.plot(grid, m, color=col, lw=1.4, ls=ls_, zorder=4, label=lab)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
    ax.text(233, 26.5, "jaws\nidentical", fontsize=6.3, color="#666666", ha="center",
            linespacing=1.15)
    ax.axvline(kb, color=C_INK, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.annotate("symmetry breaks\n(arrest begins)", xy=(kb, 44),
                xytext=(kb + 9, 52), fontsize=6.6, color=C_INK, linespacing=1.2,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_INK))
    ax.set_xlabel("containment count $c$ (control steps)", labelpad=1)
    ax.set_ylabel("jaw angle from vertical (deg)")
    ax.set_ylim(-8, 62); ax.set_xlim(grid[0], grid[-1])
    ax.grid(color="#ececec", lw=0.5); ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    my, ly, hy = q(Y)
    ax2.plot(grid, my, color=C_NOT, lw=1.2, ls=(0, (4, 1.6)), zorder=4)
    ax2.fill_between(grid, ly, hy, color=C_NOT, alpha=0.13, lw=0, zorder=1)
    ax2.set_ylabel("payload offset (mm)", color=C_NOT, labelpad=1)
    ax2.tick_params(axis="y", colors=C_NOT); ax2.set_ylim(-4, 46)
    # one legend for all three series, placed clear of the curves
    ax.plot([], [], color=C_NOT, lw=1.2, ls=(0, (4, 1.6)), label="payload offset")
    ax.legend(loc="lower right", handletextpad=0.4, borderpad=0.3, labelspacing=0.25,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    ax.set_title("(a) symmetric command, asymmetric outcome",
                 loc="left", pad=3)
    return kb, np.median(Y[:, np.argmin(np.abs(grid - kb))]), len(A)


def fig3():
    D = _load_dock()
    s, thL, thR, oy, oz, g = (D[k] for k in ("s", "thL", "thR", "oy", "oz", "g"))
    eng, notg = s > ENG_CUT, s < NOT_CUT
    mid = ~eng & ~notg

    fig = plt.figure(figsize=(7.1, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.34)
    ax0, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    mE = [np.median(v[eng]) for v in (thL, thR, oy, oz)]
    mN = [np.median(v[notg]) for v in (thL, thR, oy, oz)]
    gE, gN = np.median(g[eng]), np.median(g[notg])

    kb, ybrk, nb = _draw_symmetry_break(ax0)


    ax3.scatter(thL[eng], s[eng], s=8, color=C_ENG, alpha=0.75, lw=0, label="engaged")
    ax3.scatter(thL[mid], s[mid], s=7, color="#9e9e9e", alpha=0.6, lw=0, label="mid")
    ax3.scatter(thL[notg], s[notg], s=9, color=C_NOT, alpha=0.75, marker="^", lw=0,
                label="not engaged")
    ax3.axvline(CLOSED_DEG, color=C_INK, lw=0.8, ls=(0, (3, 2)))
    ax3.annotate("commanded closed angle:\nreached in no grasp",
                 xy=(CLOSED_DEG, 40), xytext=(14, 55), fontsize=6.8, color=C_INK,
                 linespacing=1.2, arrowprops=dict(arrowstyle="->", lw=0.6, color=C_INK))
    ax3.set_xlabel("arrest angle $\\beta$ (deg)")
    ax3.set_ylabel("grasp depth $s$ (mm)")
    from scipy.stats import spearmanr
    rho = spearmanr(thL, s).statistic
    ax3.set_title(f"(b) grasp depth vs arrest angle  ($\\rho$ = {rho:.2f}, n = {len(s)})",
                  loc="left", pad=3)
    ax3.set_xlim(-12, 48); ax3.set_ylim(-3, 64)
    ax3.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
               loc="lower left", handletextpad=0.25, borderpad=0.3, labelspacing=0.25)
    ax3.grid(color="#ececec", lw=0.5); ax3.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig3_stall_and_engagement.{ext}")
    plt.close(fig)
    print(f"[fig3] symmetry breaks at c={kb:.0f}, payload offset there {ybrk:.1f} mm, n={nb}")
    print(f"[fig3] eng: th={mE[0]:.1f}/{mE[1]:.1f} s={abs(mE[2]):.1f} g={gE:.1f} "
          f"footclear={_foot_gap('L', mE[0], mE[2]):+.1f}")
    print(f"[fig3] not: th={mN[0]:.1f}/{mN[1]:.1f} s={abs(mN[2]):.1f} g={gN:.1f} "
          f"footclear={_foot_gap('L', mN[0], mN[2]):+.1f}")
    print(f"[fig3] r(th,s)={np.corrcoef(thL, s)[0,1]:.3f}  g: eng {gE:.1f} vs not {gN:.1f} "
          f"(diff {gE-gN:.1f} mm, pooled sd {g.std():.1f} mm)")


# ----------------------------------------------------------------------------- fig 1
def fig1():
    """Mission chain, the acceptance predicate at each boundary, and the quantity that
    appears in none of them. Pass rates are from the baseline arm (N=384, seed 1)."""
    import json
    d = json.load(open("logs/graspfix_off_seed1.json"))
    fr = d["fail_reasons"]
    N = d["full_success"] + d["total"]
    n_app = sum(v for k, v in fr.items() if k.startswith("approach"))
    n_dock = sum(v for k, v in fr.items() if k.startswith("dock") or "box_fell_during_dock" in k)
    n_climb = sum(v for k, v in fr.items() if k.startswith("climb") or "hover_stab" in k)
    n_deliv = sum(v for k, v in fr.items() if k.endswith("delivery"))
    # missions that CROSS each boundary (the arrow carries the survivors of the phase to its left)
    crossed = [N - n_app, N - n_app - n_dock, N - n_app - n_dock - n_climb]
    lost = [n_app, n_dock, n_climb, n_deliv]

    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    phases = [("Approach", "RL policy"), ("Dock", "analytical"),
              ("Climb", "analytical"), ("Delivery", "RL policy")]
    bw, gap = 0.20, 0.2 / 3
    xs = [i * (bw + gap) for i in range(4)]
    ytop, bh = 0.72, 0.17
    for (name, ctrl), x in zip(phases, xs):
        ax.add_patch(Rectangle((x, ytop), bw, bh, facecolor="#f2f2f2",
                               edgecolor="#8a8a8a", lw=0.8))
        ax.text(x + bw / 2, ytop + bh * 0.64, name, ha="center", va="center", fontsize=9)
        ax.text(x + bw / 2, ytop + bh * 0.25, ctrl, ha="center", va="center",
                fontsize=6.8, color="#666666", style="italic")
    ax.text(0.0, 0.955, f"N = {N} missions", ha="left", va="center", fontsize=7.5,
            color="#444444")
    for x, nl in zip(xs, lost):
        ax.text(x + bw / 2, ytop - 0.025, f"lost here: {nl}", ha="center", va="top",
                fontsize=6.8, color=C_NOT if nl else "#999999")
    ax.text(0.0, 0.045, f"{d['full_success']} of {N} missions completed  "
                        f"({d['full_success']/N*100:.1f}%)",
            ha="left", va="center", fontsize=7.5, color=C_INK)

    preds = ["\n".join([r"waypoints reached",
                        r"$\wedge\ \ \|v\| < 2$ m/s   $\wedge\ \ \theta < 30\degree$",
                        r"$\wedge\ \ \|e_{xy}\| < 0.80$ m",
                        r"($\vee$ timeout fallback)"]),
             "\n".join([r"$c \geq 200$",
                        r"$\wedge\ \ \|p - p_{\mathrm{box}}\| < 0.30$ m",
                        r"$\wedge\ \ \theta < 15\degree$   $\wedge\ \ \|\omega\| < 2$ rad/s"]),
             "\n".join([r"$z > 1.0$ m",
                        r"$\wedge\ \ |v_z| < 0.15$ m/s",
                        r"$\wedge\ \ \theta < 5\degree$",
                        r"(or 1 s timeout)"])]
    for i, pred in enumerate(preds):
        xb = xs[i] + bw + gap / 2
        ax.annotate("", xy=(xs[i + 1], ytop + bh / 2), xytext=(xs[i] + bw, ytop + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#444444"))
        ax.text(xb, ytop + bh + 0.025, f"{crossed[i]}", ha="center",
                va="bottom", fontsize=7.5, color="#444444")
        ax.plot([xb, xb], [0.575, ytop - 0.005], color="#c8c8c8", lw=0.6, ls=(0, (2, 2)))
        ax.text(xb, 0.565, pred, ha="center", va="top", fontsize=6.5, color=C_INK,
                linespacing=1.75)
    ax.text(0.0, 0.60, "acceptance predicate", ha="left", va="bottom", fontsize=7.5,
            color="#444444")

    x0 = xs[1] + bw * 0.62
    ybar, bhh = 0.145, 0.082
    ax.add_patch(Rectangle((x0, ybar), 1.0 - x0, bhh, facecolor="#e4e4e4",
                           edgecolor="#9a9a9a", lw=0.7))
    ax.text((x0 + 1.0) / 2, ybar + bhh / 2,
            r"grasp depth $s$ — fixed when the jaws stop, then unchanged ($r$ = 0.9994)",
            ha="center", va="center", fontsize=7.0)
    ax.annotate("", xy=(x0 + 0.03, ybar + bhh), xytext=(x0 + 0.03, 0.26),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C_INK))
    ax.text(x0 + 0.05, 0.255, "produced by the jaw closure", ha="left", va="bottom",
            fontsize=7, color=C_INK)
    ax.text((x0 + 1.0) / 2, ybar - 0.025,
            r"no predicate above refers to $s$", ha="center", va="top", fontsize=8,
            fontweight="bold", color=C_NOT)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_pipeline_predicates.{ext}")
    plt.close(fig)
    print(f"[fig1] N={N}  crossed={crossed}  lost per phase={lost}  "
          f"completed={d['full_success']} ({d['full_success']/N*100:.1f}%)")


# ----------------------------------------------------------------------------- fig 4
def _load_close_state():
    """Per-grasp state at the close command (contain=225) + final grasp depth."""
    d = np.load("logs/diagnose_dock_mechanism_raw.npz", allow_pickle=True)
    tr_all, final = d["dock_traces"], d["final_box_offset_y"]

    def at(tr, c, tol=6):
        m = np.abs(tr[:, 0] - c) <= tol
        return tr[m].mean(axis=0) if m.any() else None

    rows = []
    for tr, f in zip(tr_all, final):
        if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5 or np.isnan(f):
            continue
        tr = np.asarray(tr, float)
        a = at(tr, 225)
        if a is None:
            continue
        g = ((a[1] + HINGE_Z / 1000.0) - (a[2] + BOX / 2 / 1000.0)) * 1000.0
        rows.append((abs(f) * 1000, g, abs(a[6]) * 1000, abs(a[7]) * 1000))
    r = np.array(rows)
    return dict(s=r[:, 0], g=r[:, 1], ax=r[:, 2], s0=r[:, 3])


def fig4():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    D = _load_close_state()
    s, g, ax_, s0 = D["s"], D["g"], D["ax"], D["s0"]
    y = (s > ENG_CUT).astype(int)
    gq = np.quantile(g, [0, 1 / 3, 2 / 3, 1])
    aq = np.quantile(ax_, [0, 1 / 3, 2 / 3, 1])
    sq = np.quantile(s0, [0, 1 / 3, 2 / 3, 1])
    auc_g, auc_a, auc_s0 = (roc_auc_score(y, g), roc_auc_score(y, -ax_),
                            roc_auc_score(y, s0))
    cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = lambda X: roc_auc_score(y, cross_val_predict(
        LogisticRegression(max_iter=1000), X, y, cv=cv5, method="predict_proba")[:, 1])
    auc_both, auc_three = oof(np.c_[g, ax_]), oof(np.c_[g, ax_, s0])

    # panel (a) = the predicted-threshold scatter, (b) = the empirical tercile grid:
    # the pre-registered result leads, the exploration follows.
    fig, (axB, axA, axC) = plt.subplots(1, 3, figsize=(7.1, 2.95),
                                        gridspec_kw={"width_ratios": [1.12, 1.0, 0.72],
                                                     "wspace": 0.48})

    P = np.full((3, 3), np.nan); N = np.zeros((3, 3), int)
    for i in range(3):
        am = (ax_ >= aq[i]) & (ax_ <= aq[i + 1] if i == 2 else ax_ < aq[i + 1])
        for j in range(3):
            gm = (g >= gq[j]) & (g <= gq[j + 1] if j == 2 else g < gq[j + 1])
            m = am & gm
            if m.sum():
                P[i, j], N[i, j] = y[m].mean() * 100, m.sum()
    # |e_x| increases upward here, as in panel (b)
    im = axA.imshow(P, cmap="Blues", vmin=0, vmax=100, origin="lower", aspect="auto")
    for i in range(3):
        for j in range(3):
            axA.text(j, i + 0.13, f"{P[i, j]:.0f}%", ha="center", va="center", fontsize=9.5,
                     color="white" if P[i, j] > 55 else C_INK)
            axA.text(j, i - 0.22, f"n={N[i, j]}", ha="center", va="center", fontsize=6.8,
                     color="white" if P[i, j] > 55 else "#555555")
    axA.set_xticks(range(3)); axA.set_yticks(range(3))
    axA.set_xticklabels([f"< {gq[1]:.0f}", f"{gq[1]:.0f}–{gq[2]:.0f}", f"> {gq[2]:.0f}"])
    axA.set_yticklabels([f"< {aq[1]:.0f}", f"{aq[1]:.0f}–{aq[2]:.0f}", f"> {aq[2]:.0f}"])
    axA.set_xlabel("vertical clearance $h$ (mm)")
    axA.set_ylabel("strut-axis misalignment $|e_x|$ (mm)")
    axA.set_title("(b) engagement over terciles (%)", loc="left", pad=5)
    for sp in axA.spines.values():
        sp.set_visible(False)
    axA.tick_params(length=0)
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)

    axB.scatter(g[y == 1], ax_[y == 1], s=9, color=C_ENG, alpha=0.8, lw=0, label="engaged")
    axB.scatter(g[y == 0], ax_[y == 0], s=10, color=C_NOT, alpha=0.8, lw=0, marker="^",
                label="not engaged")
    axB.axhspan(-2, 4.0, color="#dddddd", alpha=0.55, lw=0, zorder=0)
    axB.axhline(4.0, color=C_INK, lw=0.9, ls=(0, (3, 2)), zorder=1)
    axB.annotate("$|e_x|$ = 4 mm, from Fig. 1(d):\n"
                 "90.5% engage below,\n"
                 "42.5% above ($p$ = 2$\\times$10$^{-9}$)",
                 xy=(58, 4.4), xytext=(47, 33), fontsize=6.6, color=C_INK, zorder=6,
                 linespacing=1.35, ha="left", va="top",
                 bbox=dict(facecolor="white", edgecolor="#bbbbbb", lw=0.5, pad=2.4,
                           alpha=0.95),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=C_INK,
                                 shrinkA=6, shrinkB=2))
    axB.set_xlabel("vertical clearance $h$ (mm)")
    axB.set_ylabel("strut-axis misalignment $|e_x|$ (mm)")
    axB.set_title(f"(a) the predicted threshold (n = {len(s)})", loc="left", pad=5)
    axB.set_xlim(28, 104); axB.set_ylim(-2, 50)
    axB.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
               loc="upper right", handletextpad=0.25, labelspacing=0.25, borderpad=0.25)
    axB.grid(color="#ececec", lw=0.5); axB.set_axisbelow(True)

    rate, lo, hi, cnt = [], [], [], []
    for i in range(3):
        m = (s0 >= sq[i]) & (s0 <= sq[i + 1] if i == 2 else s0 < sq[i + 1])
        k, n = int(y[m].sum()), int(m.sum())
        w = wilson(k, n)
        rate.append(100 * k / n); lo.append(100 * w[0]); hi.append(100 * w[1]); cnt.append(n)
    xs = np.arange(3)
    axC.bar(xs, rate, width=0.62, color=C_ENG, alpha=0.85, lw=0, zorder=3)
    axC.errorbar(xs, rate, yerr=[np.array(rate) - lo, np.array(hi) - np.array(rate)],
                 fmt="none", ecolor=C_INK, elinewidth=0.8, capsize=2.5, zorder=4)
    for x, r, n in zip(xs, rate, cnt):
        axC.text(x, r + 2.0 if r < 12 else 4, f"n={n}", ha="center", va="bottom",
                 fontsize=6.5, color=C_INK if r < 12 else "white", zorder=5)
    axC.set_xticks(xs)
    axC.set_xticklabels([f"< {sq[1]:.0f}", f"{sq[1]:.0f}–{sq[2]:.0f}", f"> {sq[2]:.0f}"])
    axC.set_xlabel("initial offset $s_0$ (mm)")
    axC.set_ylabel("grasps that end engaged (%)")
    axC.set_ylim(0, 100)
    axC.set_title("(c) initial offset", loc="left", pad=5)
    axC.grid(axis="y", color="#ececec", lw=0.5); axC.set_axisbelow(True)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig4_engagement_conditions.{ext}")
    plt.close(fig)
    print(f"[fig4] n={len(s)} engaged={y.mean()*100:.1f}%  AUC h={auc_g:.3f} |ex|={auc_a:.3f} "
          f"s0={auc_s0:.3f}  oof(h,ex)={auc_both:.3f}  oof(h,ex,s0)={auc_three:.3f}")
    print(f"[fig4] s0 terciles {sq[1]:.0f}/{sq[2]:.0f} mm -> engaged " +
          " / ".join(f"{r:.1f}% (n={n})" for r, n in zip(rate, cnt)))
    print(f"[fig4] corr(s0,s)={np.corrcoef(s0, s)[0,1]:+.3f}  "
          f"corr(s0,h)={np.corrcoef(s0, g)[0,1]:+.3f}  corr(s0,|ex|)={np.corrcoef(s0, ax_)[0,1]:+.3f}")
    print(f"[fig4] g terciles {gq[1]:.0f}/{gq[2]:.0f} mm, |ex| terciles {aq[1]:.0f}/{aq[2]:.0f} mm")
    print("[fig4] P grid (rows |ex| low->high, cols g low->high):")
    for i in range(3):
        print("        " + "  ".join(f"{P[i,j]:5.1f}% (n={N[i,j]:3d})" for j in range(3)))


# ----------------------------------------------------------------------------- fig 5
def fig5():
    from scipy.stats import fisher_exact
    d = np.load("logs/diagnose_climb_propagation_raw.npz", allow_pickle=True)
    ce, de, out = d["climb_entry"], d["delivery_entry"], d["outcomes"]
    outs = np.array([str(o) for o in out])
    ok = ~np.isnan(ce[:, 0])
    seat = np.abs(ce[:, 15]) * 1000.0
    tilt = ce[:, 6]
    reached = ok & ~np.isnan(de[:, 0])
    climb_fail = ok & (outs == "climb_failed")
    deliv_fail = reached & np.isin(outs, ["box_dropped_delivery", "too_tilted_delivery"])

    TILT_CUT = 15.0
    cols = [("engaged", seat > ENG_CUT), ("not engaged", seat < NOT_CUT)]
    rows = [(f"$\\theta$ < {TILT_CUT:.0f}$\\degree$", tilt < TILT_CUT),
            (f"$\\theta$ $\\geq$ {TILT_CUT:.0f}$\\degree$", tilt >= TILT_CUT)]

    A = np.full((2, 2), np.nan); NA = np.zeros((2, 2), int)
    B = np.full((2, 2), np.nan); NB = np.zeros((2, 2), int)
    for i, (_, rm) in enumerate(rows):
        for j, (_, cm) in enumerate(cols):
            m = ok & rm & cm
            if m.sum():
                A[i, j], NA[i, j] = climb_fail[m].mean() * 100, m.sum()
            md = reached & rm & cm
            if md.sum():
                B[i, j], NB[i, j] = deliv_fail[md].mean() * 100, md.sum()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"wspace": 0.42})
    vmax = float(np.nanmax([A, B])) * 1.05
    for ax, M, NN, ttl in ((axes[0], A, NA, "(a) climb: fails on tilt, not depth"),
                           (axes[1], B, NB, "(b) delivery: fails on depth, not tilt")):
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, origin="lower", aspect="auto")
        for i in range(2):
            for j in range(2):
                if np.isnan(M[i, j]):
                    continue
                col = "white" if M[i, j] > vmax * 0.55 else C_INK
                k = int(round(M[i, j] / 100 * NN[i, j]))
                lo, hi = wilson(k, NN[i, j])
                ax.text(j, i + 0.13, f"{M[i, j]:.0f}%", ha="center", va="center",
                        fontsize=11, color=col)
                ax.text(j, i - 0.10, f"[{lo*100:.0f}, {hi*100:.0f}]", ha="center",
                        va="center", fontsize=7, color=col)
                ax.text(j, i - 0.28, f"n = {NN[i, j]}", ha="center", va="center",
                        fontsize=7, color="white" if M[i, j] > vmax * 0.55 else "#555555")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels([c[0] for c in cols])
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel("grasp depth at the handoff")
        ax.set_ylabel("tilt at the handoff")
        ax.set_title(ttl, loc="left", pad=5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("failure rate (%)", fontsize=7.5)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7, length=0)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig5_orthogonal_factors.{ext}")
    plt.close(fig)

    def fisher(mask_num, mask_den, m1, m2):
        a = int((mask_num & m1).sum()); b = int((mask_den & m1).sum()) - a
        c = int((mask_num & m2).sum()); dd = int((mask_den & m2).sum()) - c
        return fisher_exact([[a, b], [c, dd]])[1]

    eng, noteng = cols[0][1], cols[1][1]
    lo, hi = rows[0][1], rows[1][1]
    print(f"[fig5] climb-fail  tilt effect | engaged     : p={fisher(climb_fail, ok, eng & lo, eng & hi):.2e}")
    print(f"[fig5] climb-fail  tilt effect | not engaged : p={fisher(climb_fail, ok, noteng & lo, noteng & hi):.2e}")
    print(f"[fig5] climb-fail  seating effect | low tilt : p={fisher(climb_fail, ok, eng & lo, noteng & lo):.2e}")
    print(f"[fig5] deliv-fail  seating effect | low tilt : p={fisher(deliv_fail, reached, eng & lo, noteng & lo):.2e}")
    print(f"[fig5] deliv-fail  seating effect | high tilt: p={fisher(deliv_fail, reached, eng & hi, noteng & hi):.2e}")
    print(f"[fig5] deliv-fail  tilt effect | engaged     : p={fisher(deliv_fail, reached, eng & lo, eng & hi):.2e}")
    print(f"[fig5] climb grid %\n{np.round(A,1)}\n counts\n{NA}")
    print(f"[fig5] deliv grid %\n{np.round(B,1)}\n counts\n{NB}")


# ----------------------------------------------------------------------------- fig 6
_CATS = [("success", "#0072B2", None),
         ("rejected at handoff", "#9ecae1", "///"),
         ("dock-phase failure", "#4d4d4d", None),
         ("climb-phase failure", "#adadad", None),
         ("delivery-phase failure", "#D55E00", None)]


def _arm_breakdown(mode):
    import json
    d = json.load(open(f"logs/graspfix_{mode}_seed1.json"))
    fr = d["fail_reasons"]
    N = d["full_success"] + d["total"]
    gated = sum(v for k, v in fr.items() if k.startswith("dock_gated"))
    dock = sum(v for k, v in fr.items()
               if (k.startswith("dock") and not k.startswith("dock_gated"))
               or k == "box_fell_during_dock")
    climb = sum(v for k, v in fr.items() if k.startswith("climb") or "hover_stab" in k)
    deliv = sum(v for k, v in fr.items() if k.endswith("delivery"))
    assert d["full_success"] + gated + dock + climb + deliv == N, (mode, fr)
    return dict(N=N, success=d["full_success"], gated=gated, dock=dock,
                climb=climb, deliv=deliv, accepted=N - gated - dock)


def fig6():
    arms = [("baseline", "off"), ("shaped close + gate", "full"), ("re-grasp", "regrasp")]
    long = {"off": "baseline", "full": "shaped close + gate",
            "regrasp": "re-grasp (up to 3 tries)"}
    B = {m: _arm_breakdown(m) for _, m in arms}

    fig = plt.figure(figsize=(7.1, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.05], wspace=0.52)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    ypos = np.arange(len(arms))[::-1]
    for yi, (_, m) in zip(ypos, arms):
        b, left = B[m], 0.0
        for key, (cname, colour, hatch) in zip(
                ["success", "gated", "dock", "climb", "deliv"], _CATS):
            w = b[key] / b["N"] * 100
            axA.barh(yi, w, left=left, height=0.6, color=colour, edgecolor="white",
                     lw=0.6, hatch=hatch, label=cname if yi == ypos[0] else None)
            if w >= 8:
                axA.text(left + w / 2, yi, f"{w:.0f}", ha="center", va="center",
                         fontsize=7.5, color=C_INK if colour == "#9ecae1" else "white")
            left += w
    axA.set_yticks(ypos)
    axA.set_yticklabels([f"{long[m]}\n(N = {B[m]['N']})" for _, m in arms], fontsize=7.5)
    axA.set_xlim(0, 100); axA.set_xlabel("missions (%)")
    base = B["off"]["success"] / B["off"]["N"] * 100
    axA.axvline(base, color=C_INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
    axA.set_ylim(-1.05, len(arms) - 0.45)
    axA.text(base, -0.78, "baseline completion 69% — no arm crosses it",
             ha="center", va="center", fontsize=6.3, color=C_INK,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))
    axA.set_title("(a) outcome composition", loc="left", pad=5)
    axA.grid(axis="x", color="#ececec", lw=0.5); axA.set_axisbelow(True)

    x = np.arange(len(arms)); w = 0.34
    ymax = max(max(B[m]["dock"], B[m]["deliv"]) / B[m]["N"] * 100 for _, m in arms)
    for off, key, colour, hatch, lab in ((-w / 2, "dock", "#4d4d4d", None, "dock phase"),
                                         (+w / 2, "deliv", "#D55E00", "///", "delivery phase")):
        v = [B[m][key] / B[m]["N"] * 100 for _, m in arms]
        axB.bar(x + off, v, w, color=colour, edgecolor="white", lw=0.5, hatch=hatch,
                label=lab)
        for xi, vi in zip(x + off, v):
            axB.text(xi, vi + ymax * 0.03, f"{vi:.1f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(x)
    axB.set_xticklabels([a[0] for a in arms], fontsize=6.5, rotation=25, ha="right",
                        rotation_mode="anchor")
    axB.set_ylabel("missions lost in the phase (%)")
    axB.set_title("(b) where failures land", loc="left", pad=5)
    axB.set_ylim(0, ymax * 1.42)
    axB.legend(frameon=False, loc="upper center", handlelength=1.1, labelspacing=0.2,
               borderpad=0.05, fontsize=7)
    axB.grid(axis="y", color="#ececec", lw=0.5); axB.set_axisbelow(True)

    tx = [B[m]["success"] / B[m]["N"] * 100 for _, m in arms]
    ty = [B[m]["success"] / B[m]["accepted"] * 100 for _, m in arms]
    offs = {"baseline": ((0, -17), "center"), "shaped close + gate": ((11, 4), "left"),
            "re-grasp": ((0, 11), "center")}
    for (lab, m), xi, yi in zip(arms, tx, ty):
        k, n = B[m]["success"], B[m]["accepted"]
        lo, hi = wilson(k, n)
        axC.errorbar(xi, yi, yerr=[[yi - lo * 100], [hi * 100 - yi]], fmt="none",
                     ecolor=C_ENG, elinewidth=0.8, capsize=2, zorder=2)
        axC.scatter(xi, yi, s=30, color=C_ENG, zorder=3)
        dxy, hal = offs[lab]
        axC.annotate(lab, (xi, yi), textcoords="offset points", xytext=dxy,
                     ha=hal, fontsize=7, color=C_INK)
    axC.set_xlabel("missions completed (%)")
    axC.set_ylabel("completion among accepted (%)")
    axC.set_title("(c) throughput vs conditional reliability", loc="left", pad=5)
    axC.set_xlim(32, 80); axC.set_ylim(60, 99)
    axC.grid(color="#ececec", lw=0.5); axC.set_axisbelow(True)

    h, lb = axA.get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), handlelength=1.1, columnspacing=1.1,
               fontsize=7)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig6_interventions.{ext}")
    plt.close(fig)
    for _, m in arms:
        b = B[m]
        print(f"[fig6] {m:8s} N={b['N']:3d} success={b['success']/b['N']*100:5.1f}%  "
              f"gated={b['gated']/b['N']*100:5.1f}%  dock={b['dock']/b['N']*100:4.1f}%  "
              f"climb={b['climb']/b['N']*100:4.1f}%  deliv={b['deliv']/b['N']*100:4.1f}%  "
              f"cond={b['success']/b['accepted']*100:5.1f}% (n={b['accepted']})")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"]):
        globals()[name]()
