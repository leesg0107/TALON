"""Reconstruct the grasp contact geometry per mission (no GPU, no shapely).

Question: when the closing plate stalls, WHERE does the box touch the gripper?
Cross-section in the body-frame Y-Z plane (the plate-closing plane).

Geometry from assets/gripper_drone_v5.urdf:
  hinge L  = (y=+20,  z=-22.5) mm, rotation about +X by theta_L
  hinge R  = (y=-20,  z=-22.5) mm, rotation about -X by theta_R
  foot L   = local (y=-15, z=-116) mm, 42 (Y) x 15 (Z)   <- spans x -50..+50, CAN touch the box
  foot R   = local (y=+15, z=-116) mm, 42 (Y) x 15 (Z)
  legs     = x = +-50 mm, 12 mm thick  -> outboard of the 80 mm box (x -40..+40): never contact
  gripper reference point (offsets measured from here) = (0, 0, -80) mm

Box: 80 mm cube, centre at (off_y, off_z - 80) in body frame, level in WORLD, so in body
frame it appears rotated by -roll. Only total tilt is logged, so we bracket roll in
{0, +tilt, -tilt} and report the range.
"""
import numpy as np

MM = 1000.0
FOOT_W, FOOT_T = 42.0, 15.0
BOX = 80.0
HINGE = {"L": np.array([+20.0, -22.5]), "R": np.array([-20.0, -22.5])}
FOOT_LOCAL = {"L": np.array([-15.0, -116.0]), "R": np.array([+15.0, -116.0])}
REF_Z = -80.0


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def foot_poly(side, theta):
    """Foot rectangle corners in body frame (y,z), mm. theta = joint angle (rad)."""
    sign = 1.0 if side == "L" else -1.0          # right joint axis is -X
    R = rot(sign * theta)
    corners = np.array([[-FOOT_W / 2, -FOOT_T / 2], [FOOT_W / 2, -FOOT_T / 2],
                        [FOOT_W / 2, FOOT_T / 2], [-FOOT_W / 2, FOOT_T / 2]])
    return (R @ (FOOT_LOCAL[side] + corners).T).T + HINGE[side]


def box_poly(off_y, off_z, roll):
    c = np.array([off_y, off_z + REF_Z])
    h = BOX / 2
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    return (rot(-roll) @ corners.T).T + c


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def poly_gap(P, Q):
    """Min distance between two convex polygons; negative-ish if overlapping (SAT depth)."""
    def overlaps(P, Q):
        for poly in (P, Q):
            n = len(poly)
            for i in range(n):
                e = poly[(i + 1) % n] - poly[i]
                ax = np.array([-e[1], e[0]])
                ax /= np.linalg.norm(ax) + 1e-12
                p, q = P @ ax, Q @ ax
                if p.max() < q.min() or q.max() < p.min():
                    return False
        return True
    if overlaps(P, Q):
        # penetration depth = min over axes of overlap length
        best = 1e9
        for poly in (P, Q):
            n = len(poly)
            for i in range(n):
                e = poly[(i + 1) % n] - poly[i]
                ax = np.array([-e[1], e[0]]); ax /= np.linalg.norm(ax) + 1e-12
                p, q = P @ ax, Q @ ax
                best = min(best, min(p.max() - q.min(), q.max() - p.min()))
        return -best
    d = 1e9
    for A, B in ((P, Q), (Q, P)):
        for p in A:
            n = len(B)
            for i in range(n):
                d = min(d, seg_dist(p, B[i], B[(i + 1) % n]))
    return d


def which_feature(P_box, P_foot):
    """Rough label: is the foot beside the box, under it, or overlapping its bottom corner?"""
    by0, by1 = P_box[:, 0].min(), P_box[:, 0].max()
    bz0 = P_box[:, 1].min()
    fy0, fy1 = P_foot[:, 0].min(), P_foot[:, 0].max()
    fz1 = P_foot[:, 1].max()
    y_over = min(by1, fy1) - max(by0, fy0)          # >0 => foot is (partly) under/over the box
    if y_over > 2 and fz1 > bz0 - 2:
        return f"UNDER box by {y_over:.0f}mm"
    if y_over > 2:
        return f"below box, inboard {y_over:.0f}mm"
    return f"beside box (y-gap {-y_over:.0f}mm)"


d = np.load("logs/diagnose_dock_mechanism_raw.npz", allow_pickle=True)
tr_all, final = d["dock_traces"], d["final_box_offset_y"]


def at(tr, c, tol=6):
    m = np.abs(tr[:, 0] - c) <= tol
    return tr[m].mean(axis=0) if m.any() else None


rows = []
for i, (tr, f) in enumerate(zip(tr_all, final)):
    if tr is None or len(np.shape(tr)) != 2 or len(tr) < 5 or np.isnan(f):
        continue
    tr = np.asarray(tr, float)
    b = at(tr, 320)
    if b is None:
        continue
    rows.append(dict(i=i, depth=f * MM, thL=b[4], thR=b[5], oy=b[7] * MM, oz=b[8] * MM,
                     tilt=np.radians(b[13])))

deep = sorted([r for r in rows if r["depth"] < -25], key=lambda r: r["depth"])[:4]
shal = sorted([r for r in rows if r["depth"] > -15], key=lambda r: -r["depth"])[:4]

print("Per-mission contact reconstruction at contain=320 (mm; gap<0 = interpenetration)")
print("roll bracketed over {0, +tilt, -tilt}\n")
for grp, nm in ((deep, "DEEP"), (shal, "SHALLOW")):
    print(f"--- {nm} ---")
    for r in grp:
        PB = {roll: box_poly(r["oy"], r["oz"], roll) for roll in (0.0, r["tilt"], -r["tilt"])}
        out = []
        for side in ("L", "R"):
            PF = foot_poly(side, r["thL"] if side == "L" else r["thR"])
            gaps = [poly_gap(PB[k], PF) for k in PB]
            out.append(f"foot{side}: gap {min(gaps):+6.1f}..{max(gaps):+6.1f}  "
                       f"[{which_feature(PB[0.0], PF)}]")
        print(f"  m{r['i']:3d} depth={r['depth']:+6.1f} thL={np.degrees(r['thL']):5.1f}deg "
              f"thR={np.degrees(r['thR']):5.1f}deg tilt={np.degrees(r['tilt']):4.1f}deg oz={r['oz']:+6.1f}")
        for o in out:
            print(f"        {o}")

# population summary: distance from each foot to the box, deep vs shallow
print("\n=== population: min foot-box gap (roll=0), mm ===")
for lo, hi, nm in ((-100, -25, "DEEP   (<-25)"), (-15, 0, "SHALLOW(>-15)")):
    gl, gr = [], []
    for r in rows:
        if not (lo <= r["depth"] <= hi):
            continue
        PB = box_poly(r["oy"], r["oz"], 0.0)
        gl.append(poly_gap(PB, foot_poly("L", r["thL"])))
        gr.append(poly_gap(PB, foot_poly("R", r["thR"])))
    if gl:
        print(f"  {nm}: n={len(gl)}  footL gap median={np.median(gl):+6.1f}  "
              f"footR gap median={np.median(gr):+6.1f}")
