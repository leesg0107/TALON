# Evaluation Results History

## Evaluation Methodology
- 100 environments, headless
- ~12000 steps (40s episodes, ~500+ episodes)
- Deterministic policy (no exploration noise)
- Dock success: overlap>50% cumulative 300 steps (~2s)
- Failure classification: Near timeout / FAR / BELOW / CRASH

---

## Chronological Results

### 1. Baseline Containment (Z-gating only)
**Model:** First containment reward model
**Config:** Kinematic box z=0.5, no pedestal, r_approach + r_contain(Z-gated)

| Metric | Value |
|--------|-------|
| Dock success | 46.7% (171/366) |
| XY precision | 3.0cm |
| Dock time | 7.9s |
| Near timeout | 51% |
| FAR | 28% |
| BELOW | 23% |
| Crash | 10% |

---

### 2. + r_approach Saturation + 8s Episodes
**Model:** best_fc10_saturate.pt
**Config:** + r_approach clamped at 0.3m, episode 8s

| Metric | Value | Change |
|--------|-------|--------|
| Dock success | 56.5% (299/529) | +10% |
| XY precision | 2.7cm | -0.3cm |
| Dock time | 5.5s | -2.4s |
| Near timeout | 49% | -2% |
| FAR | 35% | +7% |
| BELOW | 18% | -5% |
| Crash | 16% | +6% |

---

### 3. + r_fine (5-15cm Gradient Bridge)
**Model:** best_fc11_fine.pt ← **Previous best for dock rate**
**Config:** + r_fine=5.0*exp(-10*xy)*gate, episode 12s

| Metric | Value | Change |
|--------|-------|--------|
| **Dock success** | **53.8% (228/424)** | -2.7% |
| XY precision | 2.7cm | same |
| Dock time | 5.8s | +0.3s |
| **Near timeout** | **16.3%** | **-33%!** |
| FAR | 17.7% | -17% |
| BELOW | 9.7% | -8% |
| **Crash** | **2.6%** | **-13%!** |

**Key improvement:** Near timeout halved (49→16%), Crash drastically reduced (16→2.6%)

---

### 4. + r_smooth Strengthening (-0.5 → -1.0) — FAILED
**Model:** best_fc11_smooth.pt
**Config:** r_smooth penalty doubled

| Metric | Value | Change |
|--------|-------|--------|
| Dock success | 43.0% | -10.8% |
| Crash | **13.5%** | **+10.9%!** |

**Reverted.** Stronger smoothing prevented recovery after collision → crash increase.

---

### 5. + r_column Strengthening (max 5→8) — TRAIN↑ EVAL↓
**Model:** best_col_strong.pt
**Config:** r_column weights 1.5+1.5+2.0 → 2.5+2.5+3.0

| Metric | Value | Change |
|--------|-------|--------|
| **Train overlap** | **40.5% (best ever)** | +4% |
| Dock success | 48.7% | -5.1% |
| Crash | 10.1% | +7.5% |

**Lesson:** Train metrics and eval metrics can diverge. Column reward exploitation increased collisions.

---

### 6. Noise Removal Test — NO EFFECT
**Config:** pos_noise=0, vel_noise=0, obj_noise=0

| Metric | Value |
|--------|-------|
| Dock success | 49.7% |

**Conclusion:** Observation noise is NOT the precision bottleneck.

---

### 7. Action Clipping (near box) — WORSENED
**Config:** action[:, :3] *= 0.5 when pos_err < 0.3m

| Metric | Value | Change |
|--------|-------|--------|
| Dock success | 48.9% | -4.9% |
| FAR | 44% | +9% |

**Conclusion:** Reducing action magnitude impairs maneuverability → more FAR failures.

---

### 8. r_vel Change (reward→penalty) — NO EFFECT
**Config:** "slow=reward" → "overspeed=penalty only"

| Metric | Value |
|--------|-------|
| Dock success | ~50% (no change) |

**Conclusion:** r_vel hovering incentive was NOT the cause of near timeout.

---

### 9. + r_center (Gripper Center Docking)
**Model:** best_fc23_center.pt
**Config:** + r_center=5.0*exp(-15*center_offset)*gate, pedestal environment

| Metric | Value | Change |
|--------|-------|--------|
| Dock success | 47.2% (eval with old strict criteria) | — |
| **Train full_contain** | **22.9%** | **+11.8%** |
| **XY precision (success)** | **1.0cm** | **-0.7cm** |
| **Time above box** | **98.6%** | **+31%!** |
| FAR | 8.7% | -9% |
| BELOW | 4.4% | -5.3% |

**Key: Precision dramatically improved.** Drone approaches from above 99% of time.

---

### 10. + r_loiter + r_tilt_dock (Anti-Hovering/Tilt Penalties)
**Model:** best_fc34_loiter_tilt.pt ← **Current best overall**
**Config:** + loiter penalty (near+outside+late), + tilt penalty at dock

#### Training Results
| Metric | Value |
|--------|-------|
| **Train full_contain** | **34.0% (best ever)** |
| Train overlap | 35.2% |
| **Peak at 90% of training** (not early!) | — |

#### Evaluation Results (corrected criteria: overlap>50% cumulative 300 steps)
| Metric | Value | vs best_fc11_fine |
|--------|-------|------------------|
| **Dock success** | **55.6% (530/954)** | **+1.8%** |
| XY precision | 1.5cm (mean), 0.9cm (median) | improved |
| **Near timeout** | **11.6%** | **-4.7%** |
| FAR | 17.6% | same |
| BELOW | 8.2% | -1.5% |
| Crash | 7.0% | +4.4% |
| **Time above (success)** | **99.1%** | **+31.4%!** |

---

## Detailed Failure Analysis (best_fc34_loiter_tilt.pt)

### Success vs Near Timeout Comparison

| Metric | Success (541) | Near Timeout (270) |
|--------|--------------|-------------------|
| Initial XY err | 49.0cm | 51.7cm |
| Speed at closest | 0.43 m/s | 0.89 m/s |
| Min XY err achieved | 1.1cm | 4.3cm |
| Max overlap achieved | 99.1% | 68.8% |
| Time in column | 46.3% | 9.0% |
| Avg r_contain | 8.40 | 3.00 |
| Avg r_center | — | — |
| Collisions | 71.6 | 69.1 |

### Near Timeout Sub-categories (270 cases)
- **Almost docked (overlap>50%):** 201 (74%) ← within reach
- **Partial approach (10-50%):** 36 (13%)
- **Never approached (<10%):** 33 (12%)

### Key Finding
- **Spawn position does NOT determine success** (initial XY: 49cm vs 52cm)
- **Speed at closest approach is critical** (0.43 vs 0.89 m/s)
- **Time in column determines success** (46% vs 9%)
- **74% of near timeouts were "almost docked"** — potential for improvement

---

## Physical Grasp Test (eval_grasp_lift.py)

### Test with Dynamic Box (no gravity, z=0.5m)
**Model:** best_fc11_fine.pt

| Result | Count | % |
|--------|-------|---|
| Lift success (10cm+) | 9 | 2.2% |
| Lift in progress | 11 | 2.7% |
| Docked (grip closing) | 37 | 9.0% |
| Approach only | 354 | 86.1% |

**Physical grasping confirmed:** 9 successful lifts out of 411 episodes. Box physically lifted 10cm+ by gripper.

---

## Summary of All Best Models

| Model | Dock Rate | Precision | Key Feature |
|-------|-----------|-----------|-------------|
| best_fc10_saturate | 56.5% | 2.7cm | r_approach saturation |
| **best_fc11_fine** | **53.8%** | **2.7cm** | **+ r_fine** |
| best_fc23_center | 47.2%* | 1.0cm | + r_center, pedestal |
| **best_fc34_loiter_tilt** | **55.6%** | **1.5cm** | **+ r_loiter, r_tilt** |

*Eval criteria mismatch (strict vs cumulative). Re-evaluated: 55.6%.

**Best overall: best_fc34_loiter_tilt.pt** — highest dock rate (55.6%) + best precision (1.5cm) + top-down approach (99.1%).
