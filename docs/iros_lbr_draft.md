# IROS 2026 LBR Draft — Sections I–IV

> **Target**: IROS 2026 Late Breaking Results (2 pages, IEEE format)
> **Deadline**: July 31, 2026
> **Note**: LBR = compressed format. Related Work absorbed into Introduction. System Design + Method merged into Section II.

---

## Title

**Curriculum Reinforcement Learning for Autonomous Aerial Grasping with Dual-Purpose Landing Gear**

---

## I. INTRODUCTION

Aerial manipulation—the ability of unmanned aerial vehicles (UAVs) to physically interact with objects—has broad applications in logistics, inspection, and disaster response.
Conventional approaches mount a dedicated gripper or robotic arm beneath the airframe [1], [2], which adds significant payload mass and shifts the center of gravity, degrading flight efficiency and requiring larger propulsion systems.

An alternative line of research repurposes the landing gear itself as a grasping mechanism [3]–[5].
Because landing gear is structurally required for takeoff and landing yet remains idle during flight, converting it into a gripper achieves dual-purpose functionality with minimal added actuation.
Prior dual-use designs, however, rely on hand-crafted control logic or open-loop grasping sequences, limiting their adaptability to varied objects and approach conditions.

Reinforcement learning (RL) offers a path toward adaptive whole-body control of such systems. Recent work has demonstrated RL-based quadrotor flight [6] and aerial grasping with dedicated manipulators [7], but the intersection—learning autonomous grasping with a dual-purpose landing gear mechanism—remains unexplored.

In this letter, we present a simulation study of a quadrotor whose landing gear doubles as a two-finger parallel gripper, trained end-to-end via curriculum RL. Our contributions are:

- A minimal dual-purpose mechanism (1-DOF revolute $\times$ 2) that transitions between landing and grasping modes with no additional end-effector.
- A five-stage curriculum that decomposes the aerial grasping task into progressively harder sub-tasks, enabling stable learning of the full approach-grasp-transport pipeline.
- A simulation framework in Isaac Lab with 4,096 parallel environments, including a hierarchical controller (PD attitude inner loop at 300 Hz + RL outer loop at 150 Hz) and physics-based grasp verification.

## II. SYSTEM AND METHOD

### A. Platform and Dual-Purpose Mechanism

The simulated platform is a 1.08 kg quadrotor in X-configuration (arm length 175 mm, propeller radius 65 mm).
Beneath the airframe, two symmetric plates are each attached via a revolute joint about the body X-axis (range $[-5°, 50°]$, effort limit 8 N$\cdot$m). In *landing mode* the plates splay outward at 45° to form a stable landing base; in *grasping mode* the policy commands them inward to enclose an object (Fig. 1). The entire mechanism adds only 110 g (two plates at 55 g each) with no additional actuators beyond the two joint servos already embedded in the landing struts.

Mass budget: body 650 g, arms $4 \times 25$ g, motors $4 \times 55$ g, plates $2 \times 55$ g = 1.08 kg total. Inertia tensors, friction coefficients ($\mu = 1.2$), and joint dynamics (stiffness 40 N$\cdot$m/rad, damping 5 N$\cdot$m$\cdot$s/rad) are derived from CAD geometry and applied in the URDF model.

### B. Simulation Environment

We use NVIDIA Isaac Lab (v2.3.2) with Isaac Sim 5.1.0 for GPU-parallel physics simulation. The environment runs at 300 Hz; the RL policy queries at 150 Hz (decimation = 2). Up to 4,096 environments execute in parallel. Domain randomization is applied per-episode: mass $\times[0.9, 1.1]$, motor thrust coefficient $\times[0.85, 1.15]$, wind (0.5 N std, sinusoidal at 0.5 Hz), and sensor noise on position (0.02 m) and velocity (0.05 m/s).

### C. Control Architecture

The system employs a two-level control hierarchy.

**Inner loop (300 Hz).** A PD attitude controller based on SO(3) error [8] converts the RL policy's acceleration and body-rate commands into per-motor thrust via a $4 \times 4$ mixing matrix. First-order motor dynamics ($\tau = 20$ ms) model actuator lag. The controller maintains stable flight while the outer loop focuses on task-level decisions.

**Outer loop (150 Hz).** A PPO agent [9] outputs an 8-D action:
$$\mathbf{a} = [\mathbf{a}_{\text{lin}} \in \mathbb{R}^3,\; \boldsymbol{\omega}_{\text{cmd}} \in \mathbb{R}^3,\; \psi_{\text{ref}},\; \theta_{\text{grip}}]$$
where $\mathbf{a}_{\text{lin}}$ is desired body-frame linear acceleration ($\pm 8$ m/s$^2$), $\boldsymbol{\omega}_{\text{cmd}}$ is desired body angular rate ($\pm 3$ rad/s), $\psi_{\text{ref}}$ is yaw reference, and $\theta_{\text{grip}}$ is the target plate angle.

The observation is a 31-D vector:
$$\mathbf{o} = [\mathbf{v}_b,\; \boldsymbol{\omega}_b,\; \text{vec}(\mathbf{R}),\; \mathbf{p}^b_{\text{goal}},\; \mathbf{e}_{\text{pos}},\; \mathbf{a}^{xy}_{\text{prev}},\; \boldsymbol{\theta}_{\text{plates}},\; \mathbf{p}^b_{\text{obj}},\; g,\; \hat{m},\; d_{\text{obj}}]$$
comprising body-frame velocity (3), angular velocity (3), rotation matrix (9), goal in body frame (3), position error in world frame (3), previous XY action (2), normalized plate angles (2), object position in body frame (3), grasp flag (1), payload estimate (1), and object distance (1). This unified representation is used across all stages; unused channels carry default values in early stages.

### D. Curriculum Learning

The task is decomposed into five sequential stages, each initialized from the previous stage's checkpoint:

| Stage | Task | Key Reward Terms |
|-------|------|-----------------|
| 1 | Waypoint navigation | Position (exp kernel), velocity (proximity-gated), level flight |
| 2 | Precision descent to object | Dual-scale XY alignment, adaptive descent speed, below-target penalty |
| 3 | Approach + grasp | Stage 2 rewards + continuous gripper target tracking, grasp bonus, hold reward |
| 4 | Loaded flight to delivery | Position, angular stability, hold, gentle acceleration |
| 5 | Release at delivery + ascend | Approach, release bonus at target, ascend reward |

**Reward design.** The core reward primitive is a negative exponential kernel $r = w \cdot \exp(-\alpha \cdot e)$, where $e$ is the task error. This provides dense gradient near zero error while naturally saturating for large errors. For quantities where the exponential gradient vanishes (e.g., action magnitude $\|\mathbf{a}\| > 1$), proportional penalties $r = -w \cdot e^2$ are used instead.

**Grasp detection.** No teleportation or auto-attachment is used. The policy must physically close the plates around the object. A grasp is registered when: (i) gripper-to-object distance $< 0.15$ m, (ii) plate angle $< 0.1$ rad, and (iii) lateral offset $< 0.08$ m. PhysX contact forces maintain the hold; a drop is detected if the object drifts beyond 0.2 m.

**Network.** Both policy and value networks use a 3-layer MLP (512–256–128, ELU), with orthogonal initialization (gain $\sqrt{2}$ for hidden layers, 0.01 for the policy output layer). PPO hyperparameters: $\gamma = 0.99$, $\lambda = 0.95$, $\epsilon_{\text{clip}} = 0.2$, learning rate $3 \times 10^{-4}$ (Stages 1–2) / $1.5 \times 10^{-4}$ (Stages 3+), entropy coefficient 0.005 / 0.01.

## REFERENCES (placeholder)

[1] Aerial manipulation survey — Ruggiero et al., IEEE RAM, 2018.
[2] Soft aerial gripper — Fishman et al., Soft Robotics, 2021.
[3] Landing gear as gripper — MDPI Drones, 2025.
[4] Compliant landing gear grasping — RoboSoft, 2024.
[5] Soft Drone — Kim et al., Science Robotics, 2021.
[6] Sun et al., Learning quadrotor dynamics, ICRA 2026.
[7] Swooper — aerial grasping RL, 2025.
[8] Lee et al., Geometric tracking control of a quadrotor, CDC 2010.
[9] Schulman et al., Proximal Policy Optimization Algorithms, 2017.
