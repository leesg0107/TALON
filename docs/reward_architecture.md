# Reward Architecture for Aerial Grasping with Dual-Purpose Landing Gear

## Overview

This document describes the complete reward structure, penalty design, and training environment for the gripper-drone aerial grasping system. The reward architecture is designed around three axes of objectives with targeted penalties, all operating independently (sum, never product/scaling).

## System Configuration

### Drone
- Mass: 1.08kg
- Gripper reference point: -0.08m below body center (plates midpoint)
- Gripper plates: open at 50° (0.873 rad), stiffness=300, damping=15, effort=50
- Hooks on landing gear struts for physical grasping
- 8D action space: [ax, ay, az, wx, wy, wz, yaw, gripper]
- 31D observation space (gripper-centric)

### Box (Target Object)
- Size: 8×8×8 cm cube
- Mass: 0.2 kg
- Kinematic during training (fixed, doesn't move when hit)
- Positioned on pedestal at z=0.54m (pedestal top=0.50m + box_half=0.04m)
- Friction: static=2.0, dynamic=2.0

### Pedestal
- Size: 30×30×50 cm
- Kinematic (fixed)
- XY position matches box (co-located)

### Training Parameters
- Episode length: 12-15 seconds
- Drone spawn: z=1.5m with random XY offset
- Box spawn: random XY within ±0.5m, z=0.54m (on pedestal)
- Min altitude: 0.10m
- Max tilt: 60°

---

## Three-Axis Reward Structure

### Axis 1: Approach (Coarse Navigation)

**Goal:** Guide drone from spawn toward box location.

#### r_approach — Distance-based approach with saturation

$$r_{\text{approach}} = 4.0 \cdot \exp\left(-1.2 \cdot \max(d_{\text{pos}}, 0.3)\right)$$

where $d_{\text{pos}} = \|\mathbf{p}_{\text{gripper}} - \mathbf{p}_{\text{box}}\|_2$ is the 3D gripper-to-box distance.

**Key design: Saturation at 0.3m.** For $d_{\text{pos}} < 0.3$m, reward is constant (2.79/step). This eliminates hovering incentive near box — drone gets no additional approach reward by staying close without docking.

| Distance | r_approach |
|----------|-----------|
| 1.0m | 1.21 |
| 0.5m | 2.20 |
| 0.3m | 2.79 (saturated) |
| 0.1m | 2.79 (same) |
| 0.0m | 2.79 (same) |

---

### Axis 2: Column Alignment (Top-Down Approach)

**Goal:** Guide drone to approach box from above, within the capture column defined by open gripper plates.

The capture column is the vertical projection of the gripper opening:
- X: ±0.05m (strut spacing)
- Y: ±0.104m (plate tips at 50° open angle)
- Extends downward from gripper

#### Soft Column Score

$$x_{\text{contain}} = \text{clamp}\left(1 - \frac{|x_{\text{local}}|}{0.05}, 0, 1\right)$$

$$y_{\text{contain}} = \text{clamp}\left(1 - \frac{|y_{\text{local}}|}{0.104}, 0, 1\right)$$

$$z_{\text{below}} = \mathbb{1}[z_{\text{local}} < 0.02]$$

where $(x_{\text{local}}, y_{\text{local}}, z_{\text{local}})$ is the box position in gripper local frame.

#### Approach Gate (Phase Gating)

$$g_{\text{approach}} = \text{clamp}\left(\frac{0.25 - d_{xy}}{0.20}, 0, 1\right)$$

where $d_{xy} = \|\mathbf{p}_{\text{gripper}}^{xy} - \mathbf{p}_{\text{box}}^{xy}\|_2$. This ensures column rewards are only active when drone is within 0.25m XY of box (0 outside, linear ramp to 1.0 at 0.05m).

#### Column Rewards (Axis-Decomposed)

$$r_{x\text{-align}} = 1.5 \cdot x_{\text{contain}} \cdot g_{\text{approach}}$$

$$r_{y\text{-align}} = 1.5 \cdot y_{\text{contain}} \cdot g_{\text{approach}}$$

$$r_{z\text{-below}} = 2.0 \cdot z_{\text{below}} \cdot g_{\text{approach}}$$

**Key design: Sum decomposition (not product).** Each axis provides independent gradient. If $y_{\text{contain}}=0$, $r_{x\text{-align}}$ still provides X-direction gradient. Product ($x \cdot y \cdot z$) would give zero gradient when any axis is zero.

| Position | $r_{x}$ | $r_{y}$ | $r_{z}$ | Total |
|----------|---------|---------|---------|-------|
| Center (0,0) | 1.50 | 1.50 | 2.00 | 5.00 |
| X edge (5cm,0) | 0.00 | 1.50 | 2.00 | 3.50 |
| Y edge (0,10cm) | 1.50 | 0.06 | 2.00 | 3.56 |
| Outside | 0.00 | 0.00 | 0.00 | 0.00 |

#### r_z — Altitude Matching

$$r_z = 2.0 \cdot \exp(-2.0 \cdot |z_{\text{gripper}} - z_{\text{box}}|)$$

---

### Axis 3: Precision Docking (Final Alignment)

**Goal:** Precise XY alignment for box to fit within gripper plates, and center alignment for stable grasp.

#### r_contain — Overlap Area Containment

Computes the intersection area between box footprint and gripper opening in gripper local frame:

$$\text{overlap}_x = \text{clamp}\left(\min(0.05, x_{\max}^{\text{box}}) - \max(-0.05, x_{\min}^{\text{box}}), 0, \infty\right)$$

$$\text{overlap}_y = \text{clamp}\left(\min(0.062, y_{\max}^{\text{box}}) - \max(-0.062, y_{\min}^{\text{box}}), 0, \infty\right)$$

$$\text{overlap}_{xy} = \frac{\text{overlap}_x \cdot \text{overlap}_y}{(2 \cdot 0.04)^2}$$

Z-gated to prevent false positives from hovering above:

$$z_{\text{range}} = \mathbb{1}[-0.12 < z_{\text{local}} < 0.02]$$

$$\text{overlap}_{\text{ratio}} = \text{overlap}_{xy} \cdot z_{\text{range}}$$

$$r_{\text{contain}} = 15.0 \cdot \text{overlap}_{\text{ratio}} + 10.0 \cdot \mathbb{1}[\text{overlap}_{\text{ratio}} > 0.50]$$

**Novel contribution:** Area-based containment reward for aerial grasping. No prior work uses projected overlap as RL reward signal.

| Alignment | overlap_ratio | r_contain |
|-----------|--------------|-----------|
| Perfect center | 1.00 | 25.0 |
| 2cm offset | ~0.75 | 11.3 |
| 5cm offset | ~0.25 | 3.8 |
| Outside | 0.00 | 0.0 |

#### r_fine — Precision Bridge (5-15cm gap)

Bridges the gradient gap between r_approach (coarse) and r_contain (fine):

$$r_{\text{fine}} = 5.0 \cdot \exp(-10.0 \cdot d_{xy}) \cdot g_{\text{approach}}$$

| XY distance | r_fine |
|-------------|--------|
| 15cm | 0.56 |
| 10cm | 1.38 |
| 5cm | 3.03 |
| 1cm | 4.52 |

#### r_center — Center Docking (Prevents Skewed Landing)

Rewards alignment of box center with gripper center in local frame:

$$d_{\text{center}} = \|(x_{\text{local}}, y_{\text{local}})\|_2$$

$$r_{\text{center}} = 5.0 \cdot \exp(-15.0 \cdot d_{\text{center}}) \cdot g_{\text{approach}}$$

**Key design:** Operates in gripper LOCAL frame, not world frame. Penalizes skewed docking where overlap may be high but box is off-center.

| Center offset | r_center |
|---------------|----------|
| 0cm | 5.00 |
| 2cm | 3.70 |
| 5cm | 2.36 |
| 10cm | 1.12 |

---

## Penalties

### r_safe — Pedestal Collision Avoidance

$$r_{\text{safe}} = 2.0 \cdot \mathbb{1}[z_{\text{gripper}} > 0.50]$$

Binary: +2.0 above pedestal top, 0 below. Independent reward, does not gate other rewards.

### r_urgency — Time Pressure

$$\text{progress} = \frac{t}{T_{\max}}$$

$$r_{\text{urgency}} = -2.0 \cdot \text{progress} \cdot (1 - \mathbb{1}[\text{overlap} > 0.5])$$

- Docked (overlap>50%): penalty = 0
- Not docked, early (progress=0.2): penalty = -0.4
- Not docked, late (progress=0.8): penalty = -1.6

### r_loiter — Anti-Hovering (Near Box But Not Docking)

$$\text{near\_outside} = \mathbb{1}[d_{xy} < 0.2] \wedge \mathbb{1}[\text{column\_score} < 0.1]$$

$$r_{\text{loiter}} = -2.0 \cdot \text{near\_outside} \cdot \mathbb{1}[\text{progress} > 0.5]$$

**Three conditions must ALL be true:** near box (<0.2m) + outside column + late in episode (>50%). Approach phase and docked state are never penalized.

### r_tilt_dock — Anti-Skewed Docking

$$\theta_{\text{tilt}} = \arccos(R_{2,2})$$

$$r_{\text{tilt}} = -2.0 \cdot \theta_{\text{tilt}} \cdot \mathbb{1}[\text{overlap} > 0.3]$$

Only active near docking (overlap>30%). Encourages level approach for stable grasp.

| Tilt | Near dock | r_tilt |
|------|-----------|--------|
| 3° (0.05 rad) | Yes | -0.10 |
| 10° (0.17 rad) | Yes | -0.35 |
| 15° (0.26 rad) | Yes | -0.52 |
| Any | No | 0.00 |

---

## Stability Rewards (Always Active)

### r_vel — Proximity Slowdown

$$r_{\text{vel}} = \exp\left(-2.0 \cdot \|\mathbf{v}\| \cdot \text{clamp}(1 - d_{\text{pos}}, 0, 1)\right)$$

### r_smooth — Action Smoothness

$$r_{\text{smooth}} = -0.5 \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$$

### r_mag — Action Magnitude

$$r_{\text{mag}} = -0.1 \cdot \|\mathbf{a}_t\|^2$$

---

## Success Detection

### Training: Cumulative Containment

$$\text{is\_contained}_t = \mathbb{1}[\text{overlap}_{\text{ratio}} > 0.50]$$

$$\text{count}_t = \text{count}_{t-1} + \text{is\_contained}_t \quad \text{(cumulative, no reset)}$$

$$r_{\text{success}} = 50.0 \cdot \mathbb{1}[\text{count} \geq 450] \quad \text{(~3 seconds cumulative)}$$

Episode truncation on success (fast reset for more practice).

### Training: Auto Gripper Close

$$\text{gripper\_cmd} = \begin{cases} 0.873 \text{ (open)} & \text{if count} < 300 \\ -0.087 \text{ (closed)} & \text{if count} \geq 300 \end{cases}$$

Gripper automatically closes after ~2 seconds cumulative containment.

### Termination Conditions
- Drone altitude < 0.10m (crash)
- Drone XY distance > 10.0m (too far)
- Drone tilt > 60° (unstable)
- Box z < 0.45m AND not docked (box fell off pedestal)

---

## Total Reward

$$R = \underbrace{r_{\text{approach}}}_{\text{Axis 1}} + \underbrace{r_{x} + r_{y} + r_{z\text{-below}} + r_z}_{\text{Axis 2}} + \underbrace{r_{\text{contain}} + r_{\text{fine}} + r_{\text{center}}}_{\text{Axis 3}} + \underbrace{r_{\text{safe}} + r_{\text{urgency}} + r_{\text{loiter}} + r_{\text{tilt}}}_{\text{Penalties}} + \underbrace{r_{\text{vel}} + r_{\text{smooth}} + r_{\text{mag}}}_{\text{Stability}} + r_{\text{success}}$$

**All components are independent — summed, never multiplied or scaled by each other.**

### Reward Budget (Approximate)

| Component | Min | Max | Typical (approach) | Typical (docked) |
|-----------|-----|-----|-------------------|------------------|
| r_approach | 0.0 | 2.79 | 2.0 | 2.79 |
| r_column (x+y+z) | 0.0 | 5.0 | 0.0 | 3.0 |
| r_contain | 0.0 | 25.0 | 0.0 | 15.0 |
| r_fine | 0.0 | 5.0 | 0.0 | 4.0 |
| r_center | 0.0 | 5.0 | 0.0 | 4.0 |
| r_z | 0.0 | 2.0 | 0.5 | 1.5 |
| r_safe | 0.0 | 2.0 | 2.0 | 2.0 |
| r_vel | 0.0 | 1.0 | 0.5 | 0.8 |
| r_urgency | -2.0 | 0.0 | -0.5 | 0.0 |
| r_loiter | -2.0 | 0.0 | 0.0 | 0.0 |
| r_tilt_dock | ~-0.5 | 0.0 | 0.0 | -0.1 |
| r_smooth | ~-1.0 | 0.0 | -0.3 | -0.2 |
| r_mag | ~-0.5 | 0.0 | -0.1 | -0.1 |
| **Total** | | | **~4.1** | **~32.7** |

Docking provides ~8x more reward than approach hovering → strong incentive to dock.

---

## Key Design Principles

1. **All rewards independent:** Sum, never product or cross-scaling. Prevents one reward from killing another's gradient.

2. **Axis decomposition:** Column rewards use x+y+z (sum) not x*y*z (product). Each axis provides gradient even when others are zero.

3. **Soft gates, never binary:** approach_gate ramps linearly from 0 to 1. No gradient cliffs.

4. **Phase gating (Swooper-inspired):** Column/precision rewards only active within 0.25m XY. Prevents exploitation from far.

5. **Saturation prevents hovering:** r_approach saturated at 0.3m. No incentive to hover near box without docking.

6. **Penalties are conditional:** r_loiter requires 3 conditions (near + outside + late). Normal approach/docking never penalized.

7. **Gripper-centric everything:** All observations and rewards measured from gripper center (-0.08m), not body. Eliminates tilt-induced offset error.

8. **Area-based containment (novel):** Overlap reward measures actual projected area intersection, not point distance. Structurally prevents side approach.

---

## Performance Summary

### Best Model: best_fc34_loiter_tilt.pt

| Metric | Value |
|--------|-------|
| Dock success rate | 55.6% (530/954 episodes) |
| Docking precision (XY) | 1.5cm (mean), 0.9cm (median) |
| Time above box (success) | 99.1% |
| Near timeout | 11.6% |
| FAR | 17.6% |
| BELOW | 8.2% |
| Crash | 7.0% |

### Training History

| Phase | Key Change | Train full_contain | Eval dock |
|-------|-----------|-------------------|-----------|
| Overlap only | Area containment | 2.4% | — |
| + r_fine | 5-15cm bridge | 11.1% | 53.8% |
| + r_approach saturate | Hovering prevention | 10.8% | 56.5% |
| + r_center | Center docking | 22.9% | 47.2% |
| + pedestal + r_safe | Realistic environment | 22.9% | 47.2% |
| + r_loiter + r_tilt_dock | Anti-hovering/tilt penalties | 34.0% | 55.6% |
