"""Mechanism figure rendered directly from assets/gripper_drone_v5.urdf.

The URDF contains only primitive solids (boxes, cylinders), so the vehicle can be
drawn exactly: every face below is a link geometry placed by forward kinematics, not
a schematic. Output: data/paper_figures/fig1_mechanism.{pdf,png}

Run:  python scripts/make_mechanism_fig.py
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon as MplPolygon, Rectangle, FancyArrowPatch

URDF = "assets/gripper_drone_v5.urdf"
OUT = "data/paper_figures"
os.makedirs(OUT, exist_ok=True)

C_ENG, C_NOT, C_INK = "#0072B2", "#D55E00", "#1a1a1a"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "text.color": C_INK, "figure.dpi": 200, "savefig.dpi": 500,
    "savefig.bbox": "tight",
})

PAYLOAD = 0.080          # m, cube
PEDESTAL_TOP = 0.500     # m, world
OPEN_DEG, CLOSED_DEG = 50.0, -5.0


# --------------------------------------------------------------------- URDF
def _xyz(s):
    return np.array([float(v) for v in s.split()]) if s else np.zeros(3)


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle_R(axis, ang):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def load_urdf(path):
    root = ET.parse(path).getroot()
    links = {}
    for L in root.findall("link"):
        geos = []
        for v in L.findall("visual"):
            o = v.find("origin")
            T = (_xyz(o.get("xyz")) if o is not None else np.zeros(3))
            R = (rpy_to_R(*_xyz(o.get("rpy"))) if o is not None and o.get("rpy") else np.eye(3))
            g = v.find("geometry")
            for kind in ("box", "cylinder", "sphere"):
                e = g.find(kind)
                if e is None:
                    continue
                if kind == "box":
                    geos.append(("box", T, R, _xyz(e.get("size"))))
                elif kind == "cylinder":
                    geos.append(("cyl", T, R, np.array([float(e.get("radius")),
                                                        float(e.get("length"))])))
        links[L.get("name")] = geos
    joints = []
    for J in root.findall("joint"):
        o = J.find("origin")
        joints.append(dict(
            name=J.get("name"), type=J.get("type"),
            parent=J.find("parent").get("link"), child=J.find("child").get("link"),
            xyz=_xyz(o.get("xyz")) if o is not None else np.zeros(3),
            rpy=_xyz(o.get("rpy")) if o is not None and o.get("rpy") else np.zeros(3),
            axis=_xyz(J.find("axis").get("xyz")) if J.find("axis") is not None else np.array([1., 0, 0])))
    return links, joints


def fk(links, joints, q):
    """Return {link: (R, t)} in base_link frame; q maps joint name -> angle (rad)."""
    pose = {"base_link": (np.eye(3), np.zeros(3))}
    changed = True
    while changed:
        changed = False
        for j in joints:
            if j["parent"] in pose and j["child"] not in pose:
                Rp, tp = pose[j["parent"]]
                Rj = rpy_to_R(*j["rpy"])
                if j["type"] == "revolute" and j["name"] in q:
                    Rj = Rj @ axis_angle_R(j["axis"], q[j["name"]])
                pose[j["child"]] = (Rp @ Rj, tp + Rp @ j["xyz"])
                changed = True
    return pose


# --------------------------------------------------------------------- solids
def box_faces(c, R, size):
    hx, hy, hz = size / 2
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    v = (R @ v.T).T + c
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [v[list(i)] for i in idx]


def cyl_faces(c, R, rl, n=16):
    r, l = rl
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = np.stack([r * np.cos(th), r * np.sin(th), np.full(n, l / 2)], 1)
    bot = np.stack([r * np.cos(th), r * np.sin(th), np.full(n, -l / 2)], 1)
    top = (R @ top.T).T + c
    bot = (R @ bot.T).T + c
    faces = [top, bot[::-1]]
    for i in range(n):
        k = (i + 1) % n
        faces.append(np.array([top[i], top[k], bot[k], bot[i]]))
    return faces


def collect(links, pose, only=None, skip=()):
    out = []
    for name, geos in links.items():
        if name not in pose or name in skip or (only and name not in only):
            continue
        R, t = pose[name]
        for kind, To, Ro, par in geos:
            c = t + R @ To
            Rw = R @ Ro
            out.append((name, box_faces(c, Rw, par) if kind == "box" else cyl_faces(c, Rw, par)))
    return out


# --------------------------------------------------------------------- render
def _shade(base_hex, normal, light=np.array([0.4, -0.7, 0.75])):
    """Flat shading: modulate a base colour by the face normal."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    lam = 0.55 + 0.45 * max(0.0, float(n @ (light / np.linalg.norm(light))))
    rgb = np.array([int(base_hex[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return tuple(np.clip(rgb * lam, 0, 1))


def _face_normal(f):
    v = np.asarray(f)
    return np.cross(v[1] - v[0], v[2] - v[0])


def iso(ax, parts, payload=None, pedestal=False, elev=20, azim=-62):
    """All solids go into ONE Poly3DCollection so matplotlib depth-sorts across parts."""
    base = {"plate_left": "#8e8e8e", "plate_right": "#8e8e8e", "base_link": "#dcdcdc"}
    faces, cols = [], []
    for name, fs in parts:
        b = base.get(name, "#cfcfcf")
        for f in fs:
            faces.append(f)
            cols.append(_shade(b, _face_normal(f)))
    if pedestal and payload is not None:
        top = payload[2] - PAYLOAD / 2
        for f in box_faces(np.array([0, 0, top - 0.011]), np.eye(3),
                           np.array([0.17, 0.17, 0.022])):
            faces.append(f); cols.append(_shade("#f0f0f0", _face_normal(f)))
    if payload is not None:
        for f in box_faces(payload, np.eye(3), np.full(3, PAYLOAD)):
            faces.append(f); cols.append(_shade("#3f92c9", _face_normal(f)))
    pc = Poly3DCollection(faces, facecolors=cols, edgecolors="#2f2f2f",
                          linewidths=0.3, zsort="average")
    ax.add_collection3d(pc)
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.62))
    lim = 0.135
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-0.20, 0.02)
    ax.set_axis_off()


# --------------------------------------------------------------------- section
def section(ax, links, pose, oy, oz, dims=True):
    """Orthographic section in the closing plane (y-z), drawn from the same solids."""
    for name in ("plate_left", "plate_right"):
        R, t = pose[name]
        for kind, To, Ro, par in links[name]:
            if kind != "box":
                continue
            c = t + R @ To
            fs = box_faces(c, R @ Ro, par)
            pts = np.vstack(fs)[:, 1:]                 # project to (y, z)
            hull = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
            ymin, ymax = pts[:, 0].min(), pts[:, 0].max()
            zmin, zmax = pts[:, 1].min(), pts[:, 1].max()
            ax.add_patch(Rectangle((ymin, zmin), ymax - ymin, zmax - zmin,
                                   facecolor="#c9c9c9" if name == "plate_right" else "#9c9c9c",
                                   edgecolor="#3a3a3a", lw=0.6, zorder=3))
    R, t = pose["base_link"]
    for kind, To, Ro, par in links["base_link"]:
        c = t + R @ To
        ax.add_patch(Rectangle((c[1] - par[1] / 2, c[2] - par[2] / 2), par[1], par[2],
                               facecolor="#ededed", edgecolor="#9a9a9a", lw=0.6, zorder=2))
    ax.add_patch(Rectangle((oy - PAYLOAD / 2, oz - PAYLOAD / 2), PAYLOAD, PAYLOAD,
                           facecolor=C_ENG, alpha=0.20, edgecolor=C_ENG, lw=1.0, zorder=4))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def dim(ax, p0, p1, label, off=0.0, vertical=False, color=C_INK, fs=7):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    if vertical:
        p0, p1 = p0 + [off, 0], p1 + [off, 0]
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->", mutation_scale=5,
                                     lw=0.6, color=color, zorder=6))
        ax.text(p0[0] - 0.004, (p0[1] + p1[1]) / 2, label, ha="right", va="center",
                fontsize=fs, color=color, zorder=7)
    else:
        p0, p1 = p0 + [0, off], p1 + [0, off]
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->", mutation_scale=5,
                                     lw=0.6, color=color, zorder=6))
        ax.text((p0[0] + p1[0]) / 2, p0[1] + 0.004, label, ha="center", va="bottom",
                fontsize=fs, color=color, zorder=7)


def main():
    links, joints = load_urdf(URDF)
    qo = {"plate_joint_left": np.radians(OPEN_DEG), "plate_joint_right": np.radians(OPEN_DEG)}
    qc = {"plate_joint_left": np.radians(CLOSED_DEG), "plate_joint_right": np.radians(CLOSED_DEG)}
    pose_o, pose_c = fk(links, joints, qo), fk(links, joints, qc)

    fig = plt.figure(figsize=(7.1, 4.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.0], hspace=-0.02, wspace=0.10)
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # (a) landing pose, jaws open
    iso(ax0, collect(links, pose_o, skip=("depth_camera_link", "forward_camera_link",
                                          "depth_camera_optical_frame",
                                          "forward_camera_optical_frame")))
    Rl, tl = pose_o["plate_left"]
    foot_c = tl + Rl @ np.array([0.0, -0.015, -0.116])
    memb_c = tl + Rl @ np.array([0.050, 0.0, -0.055])
    hinge_c = np.array([0.0, 0.020, -0.0225])
    for pt, txt, dxy in ((hinge_c, "hinge", (0.055, 0.030)),
                         (memb_c, "member", (0.052, -0.012)),
                         (foot_c, "foot", (0.030, -0.052))):
        ax0.plot([pt[0], pt[0] + dxy[0]], [pt[1], pt[1] + dxy[0]], [pt[2], pt[2] + dxy[1]],
                 color="#555555", lw=0.5)
        ax0.text(pt[0] + dxy[0], pt[1] + dxy[0], pt[2] + dxy[1], txt, fontsize=6.8,
                 color="#333333", ha="left", va="center")
    ax0.set_title("(a) open pose, jaws at 50°", loc="left", pad=-2)

    # (b) same vehicle with the payload between the jaws
    pay = np.array([0.0, 0.0, -0.0225 - 0.052 - PAYLOAD / 2])   # hinge 52 mm above top face
    iso(ax1, collect(links, pose_o, skip=("depth_camera_link", "forward_camera_link",
                                          "depth_camera_optical_frame",
                                          "forward_camera_optical_frame")),
        payload=pay, pedestal=True)
    ax1.set_title("(b) descended over the payload", loc="left", pad=-2)

    # (c) closing-plane section: open, measured arrest, and the unreachable commanded pose
    PZ = -0.0225 - 0.052 - PAYLOAD / 2
    ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    R0, t0 = pose_o["base_link"]
    for kind, To, Ro, par in links["base_link"]:
        c = t0 + R0 @ To
        ax2.add_patch(Rectangle((c[1] - par[1] / 2, c[2] - par[2] / 2), par[1], par[2],
                                facecolor="#ededed", edgecolor="#9a9a9a", lw=0.6, zorder=2))
    ax2.add_patch(Rectangle((-PAYLOAD / 2, PZ - PAYLOAD / 2), PAYLOAD, PAYLOAD,
                            facecolor=C_ENG, alpha=0.30, edgecolor=C_ENG, lw=1.1, zorder=6))
    def _hull(pts):
        pts = np.asarray(pts, float)
        c = pts.mean(0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        q = pts[np.argsort(ang)]
        keep = [q[0]]
        for pt in q[1:]:
            if np.linalg.norm(pt - keep[-1]) > 1e-9:
                keep.append(pt)
        return np.array(keep)

    for ang, fc, ec, ls, zo in [(8.0, "#8f8f8f", "#2b2b2b", "-", 4),
                                (CLOSED_DEG, "none", C_INK, (0, (3, 2)), 5)]:
        pz = fk(links, joints, {"plate_joint_left": np.radians(ang),
                                "plate_joint_right": np.radians(ang)})
        for nm in ("plate_left", "plate_right"):
            R, t = pz[nm]
            for kind, To, Ro, par in links[nm]:
                if kind != "box":
                    continue
                pts = np.vstack(box_faces(t + R @ To, R @ Ro, par))[:, 1:]
                ax2.add_patch(MplPolygon(_hull(pts), closed=True, facecolor=fc,
                                         edgecolor=ec, lw=0.7, ls=ls, zorder=zo))
    pzc = fk(links, joints, {"plate_joint_left": np.radians(CLOSED_DEG),
                             "plate_joint_right": np.radians(CLOSED_DEG)})
    ys, zs = [], []
    for nm in ("plate_left", "plate_right"):
        R, t = pzc[nm]
        for kind, To, Ro, par in links[nm]:
            if kind == "box" and abs(par[0] - 0.100) < 1e-6:
                pts = np.vstack(box_faces(t + R @ To, R @ Ro, par))[:, 1:]
                ys += [pts[:, 0].min(), pts[:, 0].max()]; zs.append(pts[:, 1].min())
    span = max(ys) - min(ys)
    dim(ax2, (min(ys), min(zs) - 0.026), (max(ys), min(zs) - 0.026),
        f"feet cover {span*1000:.0f} mm", color=C_NOT)
    dim(ax2, (-PAYLOAD / 2, PZ - PAYLOAD / 2 - 0.048), (PAYLOAD / 2, PZ - PAYLOAD / 2 - 0.048),
        f"payload {PAYLOAD*1000:.0f} mm", color=C_ENG)
    ax2.annotate("commanded\nclosed pose\n(not reachable)", xy=(0.016, -0.112),
                 xytext=(0.108, -0.070), fontsize=6.6, color=C_INK, ha="center",
                 linespacing=1.3, arrowprops=dict(arrowstyle="-", lw=0.6, color=C_INK))
    ax2.annotate("where the jaws\nactually stop\n($\\beta$ = 8°)", xy=(-0.040, -0.112),
                 xytext=(-0.108, -0.070), fontsize=6.6, color="#333333", ha="center",
                 linespacing=1.3, arrowprops=dict(arrowstyle="-", lw=0.6, color="#333333"))
    # vertical clearance h: hinge axis height above the payload's top face
    ax2.add_patch(FancyArrowPatch((0.0, -0.0225), (0.0, PZ + PAYLOAD / 2), arrowstyle="<->",
                                  mutation_scale=5, lw=0.7, color=C_INK, zorder=6))
    ax2.text(0.008, -0.0485, "$h$", fontsize=8, color=C_INK, ha="left", va="center", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
    ax2.set_xlim(-0.152, 0.152); ax2.set_ylim(-0.245, 0.012)
    ax2.set_title("(c) feet span 53 mm < payload 80 mm", loc="left", pad=2)

    # (d) strut-axis view: the member's swept band is fixed in x, so |e_x| < 4 mm makes
    #     member-payload contact geometrically impossible at ANY jaw angle.
    PZ3 = -0.0225 - 0.052 - PAYLOAD / 2
    ax3.set_aspect("equal"); ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values():
        sp.set_visible(False)

    bs = np.radians(np.linspace(CLOSED_DEG, OPEN_DEG, 60))
    z_top = np.max(-0.0225 + 0.0125 * np.sin(bs))
    z_bot = np.min(-0.0225 - 0.110 * np.cos(bs) - 0.0125 * np.sin(bs))
    b8 = np.radians(8.0)
    m_top = -0.0225 - 0.055 * np.cos(b8) + (0.0125 * np.sin(b8) + 0.055 * np.cos(b8))
    m_bot = -0.0225 - 0.055 * np.cos(b8) - (0.0125 * np.sin(b8) + 0.055 * np.cos(b8))

    def panel(x0, ex, caption, clear):
        for sgn in (-1, 1):
            xi = sgn * 0.044
            ax3.add_patch(Rectangle((x0 + min(xi, sgn * 0.056), z_bot),
                                    0.012, z_top - z_bot, facecolor="#d8d8d8",
                                    edgecolor="#9a9a9a", lw=0.5, hatch="\\\\", zorder=2))
            ax3.add_patch(Rectangle((x0 + min(xi, sgn * 0.056), m_bot),
                                    0.012, m_top - m_bot, facecolor="#8e8e8e",
                                    alpha=0.55, edgecolor="#2f2f2f", lw=0.6, zorder=3))
        col = C_ENG if clear else C_NOT
        ax3.add_patch(Rectangle((x0 + ex - PAYLOAD / 2, PZ3 - PAYLOAD / 2), PAYLOAD, PAYLOAD,
                                facecolor=col, alpha=0.28, edgecolor=col, lw=1.1, zorder=4))
        ax3.add_patch(Rectangle((x0 - 0.075, PZ3 - PAYLOAD / 2 - 0.024), 0.150, 0.024,
                                facecolor="#f2f2f2", edgecolor="#c8c8c8", lw=0.5, zorder=1))
        if clear:
            dim(ax3, (x0 + 0.040, PZ3 + 0.050), (x0 + 0.044, PZ3 + 0.050), "", color=C_INK)
            ax3.annotate("4 mm", xy=(x0 + 0.042, PZ3 + 0.050), xytext=(x0 + 0.080, PZ3 + 0.072),
                         fontsize=6.6, color=C_INK, ha="center",
                         arrowprops=dict(arrowstyle="-", lw=0.5, color=C_INK))
        else:
            ax3.add_patch(Rectangle((x0 + 0.044, PZ3 - PAYLOAD / 2), ex - 0.004, PAYLOAD,
                                    facecolor=C_NOT, alpha=0.85, edgecolor="none", zorder=6))
            ax3.annotate("bands overlap:\na member can\nblock the jaw",
                         xy=(x0 + 0.048, PZ3 + 0.008), xytext=(x0 - 0.030, PZ3 + 0.052),
                         fontsize=6.5, color=C_NOT, ha="center", linespacing=1.25,
                         arrowprops=dict(arrowstyle="-", lw=0.6, color=C_NOT))
        ax3.text(x0, PZ3 - PAYLOAD / 2 - 0.042, caption, ha="center", va="top",
                 fontsize=6.8, color=C_INK, linespacing=1.25)

    panel(-0.098, 0.000, "$|e_x|$ < 4 mm:\nbands disjoint", True)
    panel(+0.098, 0.0134, "$|e_x|$ = 13.4 mm (the mean):\nbands overlap", False)
    ax3.text(0.0, PZ3 + 0.112,
             "hatched = strut-axis band swept by a jaw member over its whole travel\n"
             "(the hinge axis is $x$, so the band does not move with the jaw angle)",
             ha="center", va="bottom", fontsize=6.5, color="#444444", linespacing=1.3)
    ax3.set_xlim(-0.200, 0.200); ax3.set_ylim(PZ3 - 0.100, PZ3 + 0.150)
    ax3.set_title("(d) strut-axis view: why 4 mm is the limit", loc="left", pad=2)

    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig1_mechanism.{ext}")
    plt.close(fig)
    print(f"[mech] closed-pose foot cover = {span*1000:.1f} mm ; payload = {PAYLOAD*1000:.0f} mm")
    print(f"[mech] member inner faces = ±44.0 mm ; payload half-width = {PAYLOAD/2*1000:.0f} mm "
          f"-> margin {44 - PAYLOAD/2*1000:.0f} mm")


if __name__ == "__main__":
    main()
