# TALON — Controller Design & Reward Structure
 
> Technical documentation for the TALON autonomous aerial grasping system.  
> Covers reinforcement learning reward design, analytical docking controller, and end-to-end pipeline integration.
 
---
 
## Table of Contents
 
1. [System Context](#1-system-context)
2. [Control Hierarchy](#2-control-hierarchy)
3. [RL Navigation & Transport (Stages 1 & 4)](#3-rl-navigation--transport-stages-1--4)
4. [RL Docking Attempt (Stage 3)](#4-rl-docking-attempt-stage-3)
5. [Analytical Docking Controller](#5-analytical-docking-controller)
6. [End-to-End Pipeline](#6-end-to-end-pipeline)
7. [Domain Randomization](#7-domain-randomization)
---
 
## 1. System Context
 
TALON is a quadrotor that uses its own landing gear as a parallel gripper to autonomously grasp and transport objects. The landing gear consists of two rigid struts and two servo-actuated plates. When open, they function as conventional landing gear; when closed, they envelop a target object.
 
**The core challenge**: the strut spacing is 100 mm and the target object is an 80 mm cube, leaving only **10 mm of clearance per side** along the strut axis (X). Along the plate axis (Y), clearance is a more relaxed 64 mm per side. This asymmetry drives nearly every design decision documented below.
 
**Mission flow**:
```
[Approach]  →  [Dock]  →  [Grip Close]  →  [Climb]  →  [Delivery]
  RL              Analytical      Auto          Analytical      RL
  Stage 1         PD controller                 PD (low-gain)   Stage 4
```
 
RL handles navigation (Approach, Delivery) where adaptability to wind and payload variation matters. An analytical controller handles docking (Dock, Climb) where geometric precision under sustained contact is critical. This decomposition emerged from empirical analysis: RL was first applied to docking and failed systematically (see [Section 4](#4-rl-docking-attempt-stage-3)).
 
---
 
## 2. Control Hierarchy
 
All controllers — RL and analytical — share a common two-layer architecture.
 
```
Outer loop (150 Hz):
  RL policy or Analytical controller
    → outputs: body-frame acceleration a_cmd ∈ ℝ³, yaw reference ψ_ref
 
Inner loop (300 Hz):
  a_cmd → Desired attitude (via differential flatness)
        → SO(3) geometric attitude controller (Lee et al. 2010)
        → Motor allocation (X-config mixer)
        → First-order motor dynamics (τ = 0.02s)
        → Rotor forces applied in PhysX
```
 
The inner loop is **never modified** across mission phases. Swapping between RL and analytical control means swapping only the outer-loop source of `a_cmd`. This guarantees that any outer-loop strategy can be tested without side effects on low-level stability.
 
---
 
## 3. RL Navigation & Transport (Stages 1 & 4)
 
**Source**: `gripper_waypoint_env.py`
 
Stage 1 (Approach) and Stage 4 (Delivery) share the same reward structure with mode-specific additions. Both are trained with PPO via SKRL across 4,096 parallel environments.
 
### 3.1 Observation & Action Spaces
 
| | Stage 1 (Approach) | Stage 4 (Delivery) |
|---|---|---|
| Observation dim | 22D | 23D |
| Action dim | 4D | 4D |
| Extra observation | — | payload mass estimate |
| Gripper state | Open (fixed) | Closed (fixed) |
 
Action space: `[a_x, a_y, a_z, yaw_ref]` in body frame, continuous. Note: the 4th component is a **yaw angle reference** (scaled to [−π, π]), not a yaw rate.
 
### 3.2 Reward Components
 
Each reward component targets a specific behavior. The magnitudes are tuned so that no single term dominates.
 
#### Primary: velocity toward goal
 
```python
goal_dir = normalize(goal_pos - drone_pos)
vel_toward_goal = dot(vel_world, goal_dir)
r_direction = 0.5 * clamp(vel_toward_goal, -3.0, 3.0)
```
 
Per-step range: [−1.5, +1.5]. This is the main shaping signal — it rewards progress toward the current waypoint at every timestep, not just arrival.
 
#### Primary: arrival bonus (time-dependent)
 
```python
arrived = (position_error < 0.3)
time_sec = steps_since_goal_assigned / 150.0
r_arrive = arrived * 10.0 / (time_sec + 0.5)
```
 
The `1 / (t + 0.5)` schedule makes faster arrival exponentially more valuable. At 0.5 s: reward = 10.0. At 2.0 s: reward = 4.0. This prevents policies from approaching waypoints slowly to maximize per-step `r_direction`.
 
#### Safety: crash penalty
 
```python
r_crash = -5.0 * (altitude_local < 0.15)
```
 
Large negative reward applied when altitude drops below 15 cm. Note: this is a reward signal only — termination is handled separately by `_get_dones()` with the same threshold.
 
#### Regularization: action smoothness
 
```python
r_smooth = -0.01 * sum((action - prev_action)²)
```
 
Penalizes jerk. Without this, policies learn bang-bang control that oscillates between acceleration limits, causing mechanical stress and altitude instability.
 
#### Regularization: angular velocity
 
```python
r_angular = -0.02 * norm(angular_vel_body)
```
 
Penalizes fast rotation. Complements `r_smooth` by targeting rotational oscillation specifically.
 
#### Safety: tilt penalty (mode-dependent threshold)
 
```python
# Approach (unloaded):
r_tilt = -3.0 * clamp(tilt_angle - 0.52, min=0)    # threshold ~30°
 
# Delivery (loaded):
r_tilt = -3.0 * clamp(tilt_angle - 0.45, min=0)     # threshold ~26°
```
 
**Why the loaded threshold is tighter**: aggressive tilting during transport shifts the payload's center of mass, risking detachment. The 4° reduction was empirically determined — below 26°, payload drop rate was <1%.
 
#### Transport-specific: overshoot penalty
 
```python
speed = norm(vel_world)
near_waypoint = (position_error < 1.0)
r_overshoot = -1.0 * near_waypoint * clamp(speed - 1.5, min=0)
```
 
**Why this exists**: without it, loaded-flight policies approach waypoints at full speed, overshoot, U-turn, and the aggressive tilt during the U-turn dislodges the payload. This term activates only within 1 m of the waypoint and only above 1.5 m/s — it doesn't penalize fast cruising in open space.
 
#### Timeout penalty
 
```python
timed_out = (steps_since_goal_assigned > 450)   # 3 s per waypoint
r_timeout = -2.0 * timed_out
```
 
Prevents policies from exploiting `r_direction` by circling near a waypoint without arriving.
 
### 3.3 Total Reward
 
```
Approach:  r_direction + r_arrive + r_crash + r_smooth + r_angular + r_tilt - 2.0 * timed_out
Delivery:  r_direction + r_arrive + r_crash + r_smooth + r_angular + r_tilt + r_overshoot - 2.0 * timed_out
```

The timeout penalty applies to **both** modes. The only difference is that Delivery adds `r_overshoot`.
 
### 3.4 Termination Conditions
 
| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| `too_low` | altitude < 0.15 m | Ground collision |
| `too_far` | XY distance > 10 m from origin | Left operational area |
| `too_tilted` | tilt > 60° | Unrecoverable attitude |
| `wp_timeout` | 3 s per waypoint (450 steps) | Stuck or circling |
| `box_dropped` | box altitude < 0.30 m (Delivery only) | Payload lost |
 
---
 
## 4. RL Docking Attempt (Stage 3)
 
**Source**: `drone_env.py`, Stage GRASPING
 
Before the analytical controller was designed, we attempted to solve the docking task with RL. This section documents that attempt — the reward structure, and the failure mechanisms that motivated the switch to analytical control.
 
### 4.1 Observation Space (31D, gripper-centric)
 
```
gripper_vel_body         (3)   — gripper linear velocity in body frame
angular_vel_body         (3)   — drone angular velocity in body frame
rotation_flat            (9)   — flattened rotation matrix (SO(3) → ℝ⁹)
goal_gripper_body        (3)   — goal position relative to gripper, in body frame
goal_error_world         (3)   — goal position error in world frame
prev_action_xy           (2)   — previous XY action (for smoothness)
plate_angles             (2)   — current servo angles of left/right plates
box_in_gripper_frame     (3)   — box center position relative to gripper center
grasp_flag               (1)   — binary: is box currently contained?
payload_estimate         (1)   — estimated payload mass
box_distance             (1)   — Euclidean distance to box center
```
 
**Design choice**: observations are gripper-centric, not drone-centric. The gripper is offset from the drone CoM, so drone-centric observations would require the policy to implicitly learn this offset. Gripper-centric observations make the docking geometry directly observable.
 
### 4.2 Action Space (8D)
 
```
body_acceleration        (3)   — [a_x, a_y, a_z] in body frame, scaled to [-8, 8] m/s²
body_rate_command        (3)   — [ω_x, ω_y, ω_z] angular rate, scaled to [-3, 3] rad/s
yaw_reference            (1)   — desired yaw angle, scaled to [-π, π] rad
gripper_command          (1)   — plate servo target, [-0.087, 0.873] rad (closed ↔ open)

Note: when lock_gripper=True (used in eval), gripper is auto-controlled by contain_hold_count.
```
 
### 4.3 Reward Structure
 
The reward was iteratively refined over 10+ design cycles. Across all configurations, the best RL result was 24.8% docking success on dynamic (physically-reactive) objects and ~50% on kinematic (fixed) objects. The reward structure below represents the final configuration used for dynamic-object training.
 
#### Approach rewards
 
```python
r_approach_xy = 3.0 * exp(-1.5 * xy_error)
r_approach_z  = 2.0 * exp(-2.0 * z_error) * above_box_flag
```
 
Dual-scale exponential: steep gradient near the target, gentle far away. The `above_box_flag` prevents rewarding descent before XY alignment.
 
#### Column alignment (gated)
 
```python
approach_gate = clamp((0.25 - xy_error) / 0.20, 0, 1)
r_x_align = 1.5 * x_containment * approach_gate
r_y_align = 1.5 * y_containment * approach_gate
```
 
**Why gated**: without `approach_gate`, the policy receives alignment reward while still far from the target, learning to align prematurely at high altitude where the box isn't reachable. The gate activates only within 25 cm XY.
 
#### Staged descent
 
```python
xy_aligned = sigmoid((0.05 - xy_error) / 0.015)
alt_target = box_z + 0.25 * (1 - xy_aligned)
r_z = 2.0 * exp(-2.0 * |gripper_z - alt_target|)
```
 
The altitude target is state-dependent: when XY is misaligned, the target is 25 cm above the box (hold position); when XY is aligned, the target drops to box height (descend). This creates a natural "align then descend" sequence.
 
#### Precision docking (overlap-based)
 
```python
r_contain   = 15.0 * overlap_ratio + 10.0 * full_contain_flag
r_dock_bonus = 10.0 * (overlap > 0.50)
r_hold       = 10.0 * clamp(contain_hold_count / 150.0, max=3.0) * is_contained
r_success    = 50.0 * (contain_hold_count >= 225)
```
 
The `r_hold` term increases linearly with sustained containment, rewarding the policy for maintaining position inside the gripper envelope. `r_success` is a large terminal bonus at 1.5 s of sustained containment.
 
#### Penalties
 
```python
r_smooth       = -0.2 * sum(Δaction²)          # 20× Stage 1 magnitude
r_magnitude    = -0.1 * sum(action²)            # penalize large commands
r_tilt_descent = -3.0 * tilt * xy_aligned       # no tilt during descent
r_tilt_dock    = -2.0 * tilt * (overlap > 0.3)  # no tilt during docking
```
 
**Why `r_smooth` is 20× larger than Stage 1**: during docking, any sudden action causes the gripper to jolt and hit the box. The policy must be exceptionally smooth near contact.
 
### 4.4 Why RL Docking Failed
 
Despite extensive reward engineering, the best RL policy achieved only 24.8% docking success on physically-reactive objects (down from ~50% on fixed objects). Three failure mechanisms were identified:
 
1. **Reward dilemma**: the policy learns to hover at ~12 cm above the box, collecting `r_approach` and `r_z` reward indefinitely. Descending risks losing all alignment reward if the box is bumped. The optimal exploitation strategy is to *not dock*.
2. **Contact velocity overshoot**: when the policy does attempt descent, it approaches at 0.8–1.4 m/s vertical velocity — 7× the safe threshold of ~0.15 m/s. The contact impulse pushes the box outside the clearance envelope before any corrective action can be taken.
3. **Curriculum transfer collapse**: policies trained on kinematic (fixed) boxes achieve ~50% docking, but performance drops to 12–25% when evaluated on dynamic (free) boxes. The policy has never experienced contact reactions during training and has no strategy for them.
These findings motivated the analytical controller documented in the next section.
 
---
 
## 5. Analytical Docking Controller
 
**Source**: `drone_env.py: _compute_analytical_base()`
 
The analytical controller is a multi-mode PD/PID system with gain scheduling. Each design decision traces directly to a failure mechanism identified in Section 4.4.
 
### 5.1 Failure-to-Design Traceability
 
| RL Failure Mechanism | Analytical Design Response |
|---|---|
| Reward dilemma → hover exploit | **Descent gate**: XY alignment verified, then descent is mandatory — no reward signal to game |
| Contact velocity 7× overshoot | **Two-stage sigmoid descent**: velocity profile reduces from 0.40 to 0.15 m/s approaching contact |
| Curriculum transfer collapse | **Distance-adaptive gains**: approach mode (high Kp) and precision mode (low Kp) are separate — no transfer between regimes |
| (Observed) Box pushed out by contact | **Dock proximity softening**: Kp drops and Kd rises at contact, creating compliance without force sensing |
| (Observed) XY oscillation stalls descent | **Hysteresis**: once descent starts, small XY deviations don't abort it |
| (Observed) Strut catches box edge | **Stuck recovery**: detects stall → pulls up → retries |
 
### 5.2 XY Position Control — Asymmetric PD with Dock Proximity Softening
 
```python
# Base gains (reflect clearance asymmetry)
Kp_x = 12.0,  Kd_x = 8.0     # X: strut axis, 10mm clearance → aggressive tracking
Kp_y = 8.0,   Kd_y = 6.0      # Y: plate axis, 64mm clearance → relaxed
 
# Dock proximity softening
# Activates only when: altitude < 4cm above box AND xy_error < 3cm
# i.e., the gripper is nearly enveloping the box
dock_proximity = sigmoid((0.04 - alt_above) / 0.015) * sigmoid((0.03 - xy_mag) / 0.01)
 
Kp_x -= 2.0 * dock_proximity    # 12 → 10
Kp_y -= 1.5 * dock_proximity    # 8  → 6.5
Kd_x += 1.0 * dock_proximity    # 8  → 9
Kd_y += 1.0 * dock_proximity    # 6  → 7
```
 
**Why soften at contact?**
 
Before contact, high Kp is needed to drive the gripper toward the target accurately. During contact, high Kp causes the gripper to "push through" position errors, applying force to the box and pushing it out of the clearance envelope. Reducing Kp and increasing Kd transitions the controller to a compliance-like mode: it stops actively pursuing the setpoint and instead dampens any motion. The box can settle within the gripper without being pushed.
 
This is not impedance control (no force sensing). It is gain-scheduled PD that approximates compliance behavior using only position/velocity feedback.
 
### 5.3 Z Control — XY-Alignment-Gated Standoff with PID
 
The vertical controller does not command a fixed descent rate. Instead, it tracks a **standoff altitude** above the box that smoothly decreases as XY alignment improves.
 
```python
# Standoff = f(XY alignment quality)
xy_coarse = sigmoid((0.10 - xy_mag) / 0.05)    # smooth gate at ~5-15cm
xy_fine   = sigmoid((0.03 - xy_mag) / 0.01)    # steep gate at ~2-4cm
 
standoff = 0.30 - 0.18 * xy_coarse - 0.10 * xy_fine
```
 
| XY error | Standoff | Behavior |
|----------|----------|----------|
| > 15 cm  | 30 cm    | Hold high — XY alignment first |
| ~ 5 cm   | 12 cm    | Begin descent — coarse alignment achieved |
| < 2 cm   | 2 cm     | Final descent — precision alignment |
 
```python
target_z = box_z + standoff
 
# PID altitude controller
Kp_z = 12.0,  Ki_z = 3.0,  Kd_z = 7.0
z_error = target_z - gripper_z
z_integral += z_error * dt    # clamped to [-1, 1]
 
az_world = Kp_z * z_error + Ki_z * z_integral - Kd_z * vel_z
```
 
**Why PID (not PD) on Z?** The integral term compensates for two sources of steady-state error: (1) mass mismatch from domain randomization (±10%), and (2) the added box mass after grasp. Without Ki, the drone sags 3–5 cm under payload, sometimes losing grip.
 
**Why gate descent on XY alignment?** This is the direct response to the RL reward dilemma. RL couldn't decide when to descend because descent risked losing alignment reward. The analytical controller removes this ambiguity: descent happens automatically and monotonically as XY alignment improves. There is no decision to make, and therefore no dilemma to exploit.
 
### 5.4 Gripper Close, Grasp Verification, and Climb
 
After descent, the controller enters a sequential post-dock procedure:
 
| contain_hold_count | Phase | Action |
|---|---|---|
| 0 – 149 | Dock descent | Normal standoff tracking; gripper open |
| 150 | Trigger close | Gripper plates servo to −5°; dock altitude recorded |
| 150 – 289 | Hold | Maintain recorded dock altitude; wait for plates to fully close |
| 290 – 324 | Hold + verify | Continue holding dock altitude; plates fully closed |
| ≥ 325 | Climb | PID to 1.5 m altitude with reduced gains |
 
**Grasp verification**: physical grasp verification happens during the Climb phase via false/fake-grasp detection (see Section 6.4). If the box doesn't follow the drone during climb, the state machine resets to the Dock phase for retry.
 
**Climb gain reduction**:
```python
# Docking:  Kp = 12, Kd = 7  — aggressive for precision
# Climbing: Kp = 6,  Kd = 5  — gentle to avoid payload shake
# Min clamp on az: -2.0 m/s²  — prevents hard deceleration
```
 
High Kp during climb causes rapid acceleration → payload oscillation → grip failure. Kp = 6 produces a smooth 0.3–0.4 m/s ascent. The min clamp prevents the controller from braking hard if it overshoots the target altitude.
 
### 5.5 Stuck Recovery
 
When the gripper catches on a box edge, descent stalls without achieving overlap. The controller detects this and executes a pullup-and-retry sequence.
 
**Detection** (must persist for 100 consecutive steps = 0.67 s):
```python
stuck = (alt_above < 0.06)         # close to box
      & (xy_mag > 0.04)            # but not centered
      & (norm(vel) < 0.12)         # and not moving
      & (contain_hold_count < 10)  # and not docked
```
 
**Recovery** (150 steps = 1.0 s):
```python
# Lift to 30 cm above box
recovery_target_z = box_z + 0.30
az = Kp_z * (recovery_target_z - gripper_z) - Kd_z * vel_z
 
# XY boost during recovery (re-center over box)
ax += 4.0 * xy_error_x - 2.0 * vel_x
ay += 4.0 * xy_error_y - 2.0 * vel_y
 
# After recovery completes:
# z_integral resets → clean approach with no accumulated error
```
 
### 5.6 Safety Floor
 
```python
az_world += 3.0 * (alt_above < -0.02)
```
 
If the gripper descends below the box top surface (collision with pedestal), a hard upward acceleration is applied regardless of other commands. This is a last-resort protection that overrides all other Z commands.
 
---
 
## 6. End-to-End Pipeline
 
**Source**: `scripts/eval_mission_headless.py`
 
### 6.1 Phase State Machine
 
```
SETTLE (0.5s) → APPROACH → DOCK → CLIMB → DELIVERY → ARRIVED → DONE
```
 
| Phase | Controller | Timeout | Transition Condition |
|---|---|---|---|
| SETTLE | None (zero action) | 1 step | Teleport state committed to physics |
| APPROACH | RL Stage 1 | 60 s | Final waypoint reached + stability check |
| DOCK | Analytical PID | 12 s | contain_hold ≥ 325 + box_dist < 0.30 |
| CLIMB | Analytical PID (low-gain) | 10 s | Altitude > 1.0 m |
| DELIVERY | RL Stage 4 | — | Final delivery waypoint reached |
| ARRIVED | RL Stage 4 (hover) | 3 s | Timer expires |
| DONE | — | — | Mission complete, env recycled |
 
### 6.2 RL → Analytical Transition (APPROACH → DOCK)
 
When the RL approach policy reaches the final waypoint (0.5 m above box), control transfers to the analytical controller. At this point:
 
- XY error is typically < 30 cm (RL got close but not precise)
- Z altitude is ~0.5 m above box
- The analytical controller's standoff system takes over: it sees xy_mag ~ 0.30, computes standoff ~ 30 cm, and holds altitude while refining XY alignment
No special handoff logic is needed — the analytical controller's standoff naturally handles whatever state the RL policy delivers.
 
### 6.3 Analytical → RL Transition (CLIMB → DELIVERY)
 
When the climb phase reaches 1.0 m altitude, control transfers directly to the RL Stage 4 policy. At the CLIMB → DELIVERY transition, the attitude controller mass is updated:
 
```python
# Update attitude controller mass to include payload
drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[eid]
env.attitude_ctrl.mass[eid] = drone_mass + 0.2   # 0.2 kg box
```

This updates the inner-loop gravity compensation for loaded flight. The physical box mass is handled by PhysX (dynamic rigid body with gravity). The Stage 4 RL policy additionally receives `payload_mass_estimate` in its observation vector.
 
### 6.4 False Grasp Detection
 
Two failure modes are detected during and after docking:
 
**False grasp** (during DOCK phase):
```python
false_grasp = (contain_hold_count >= 325)     # controller thinks grasp succeeded
            & (box_distance > 0.30)            # but box is far from gripper
```
 
This catches cases where `contain_hold_count` increments due to momentary overlap, but the box was knocked away before grip closed. On detection: reset to DOCK phase start, re-open gripper, retry.
 
**Fake grasp** (during CLIMB phase):
```python
fake_grasp = (gripper_z > pedestal_z + 0.25)  # drone has climbed 25cm
           & (box_z < pedestal_z + 0.05)       # but box is still on pedestal
```
 
The gripper closed around empty air, or the box slipped during the lift. Detected 0.2 s after grip close. On detection: descend back, re-open gripper, return to DOCK phase.
 
### 6.5 Pre-Simulated Grasp States
 
**Source**: `scripts/generate_grasp_states.py`
 
Stage 4 (Delivery) RL training requires starting from a state where the drone is airborne with a grasped payload. Generating this state by running the full approach→dock→climb sequence every episode would be prohibitively slow.
 
**Solution**: ~5,000 successful grasp states are pre-collected by running the analytical docking controller, filtering for successful climbs, and saving the resulting `(drone_state, box_state, gripper_state)` tuples. During Stage 4 training, each environment samples a random state from this bank at reset.
 
**Filtering criteria**: only states where the box altitude > 1.0 m AND box is within gripper envelope are saved. Failed docks and drops are excluded. This guarantees that Stage 4 training always starts from a valid loaded-flight state.
 
---
 
## 7. Domain Randomization
 
Applied per-episode at environment reset in both `GripperDroneEnv` and `GripperWaypointEnv`.
 
### 7.1 Physical Parameters
 
| Parameter | Distribution | Affects |
|-----------|-------------|---------|
| Total mass scale | 𝒰(0.9, 1.1) | All phases |
| Motor thrust coefficient scale | 𝒰(0.85, 1.15) | All phases |
| Wind phase | 𝒰(0, 2π) per axis | All phases |
| Wind force | 0.5 N amplitude, 0.5 Hz sinusoidal | All phases |
| Payload mass | 𝒰(0.15, 0.25) kg | Stage 4 only |
| Initial grasp state | Sampled from 4,996 pre-simulated states | Stage 4 only |
 
### 7.2 Observation Noise
 
| Sensor | Noise σ | Rationale |
|--------|---------|-----------|
| Body velocity | 0.03 m/s | Simulates IMU drift and vibration |
| Goal position | 0.01 m | Simulates localization error |
| Payload mass estimate | 0.02 kg | Simulates imperfect mass identification |
 
### 7.3 Training Configuration
 
| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (SKRL) |
| Network | MLP 2×128, ELU activation, orthogonal init |
| Parallel environments | 4,096 |
| GPU | NVIDIA RTX 4090 |
| Learning rate | 3 × 10⁻⁴ |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip range | 0.2 |
| Rollout length | 100 steps per env per update |
| Mini-batches | 4 (buffer split into 4 batches per epoch) |
| Update epochs | 5 |
| Entropy coefficient | 0.004 |
| State preprocessor | RunningStandardScaler |
| Initial log_std | 0.0 (std=1.0), min clamp at -2.0 (std=0.135) |