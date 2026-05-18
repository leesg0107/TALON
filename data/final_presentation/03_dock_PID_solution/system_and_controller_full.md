# System Architecture & Precision Docking Controller

## 1. Hierarchical Control Architecture

본 시스템은 **2-계층 제어 구조**로 구성된다:

```
[RL Policy / Analytical PID]  150 Hz  (외부 루프)
        ↓ desired accel, rates, yaw
[Attitude Controller]          300 Hz  (내부 루프)
        ↓ desired thrust + torque
[Motor Model]                  300 Hz
        ↓ per-motor forces
[Isaac Lab Physics (PhysX)]    300 Hz
```

- **외부 루프 (150 Hz)**: Task-level 제어. RL 정책 또는 analytical PID가 body-frame 가속도 명령 생성
- **내부 루프 (300 Hz)**: Attitude controller가 가속도 → 자세 → 모터 추력으로 변환
- **Decimation = 2**: 물리 시뮬레이션 2 step 당 정책 1 step

---

## 2. Quadrotor Model

### 2.1 Physical Parameters (URDF)

| 부품 | 질량 (kg) | 개수 | 소계 (kg) |
|------|----------|------|----------|
| Base link | 0.650 | 1 | 0.650 |
| Arm | 0.025 | 4 | 0.100 |
| Motor | 0.055 | 4 | 0.220 |
| Plate (gripper) | 0.055 | 2 | 0.110 |
| **총 질량** | | | **1.080** |

### 2.2 Motor Configuration (X-Config)

$$
d = 0.123744\,\text{m}, \quad L_{\text{arm}} = \sqrt{2} \cdot d = 0.175\,\text{m}
$$

| 모터 | 위치 $(x, y, z)$ [m] | 회전 방향 |
|------|---------------------|----------|
| M0 (front-right) | $(+d, +d, +0.0365)$ | CW ($-$) |
| M1 (front-left) | $(-d, +d, +0.0365)$ | CCW ($+$) |
| M2 (back-left) | $(-d, -d, +0.0365)$ | CW ($-$) |
| M3 (back-right) | $(+d, -d, +0.0365)$ | CCW ($+$) |

### 2.3 Thrust & Torque Model

$$
F_i = k_f \cdot \omega_i^2, \quad \tau_i^{\text{yaw}} = k_m \cdot \omega_i^2 \cdot s_i
$$

| 파라미터 | 값 | 단위 |
|---------|-----|------|
| $k_f$ | $1.0 \times 10^{-5}$ | $\text{N}/(\text{rad/s})^2$ |
| $k_m$ | $1.6 \times 10^{-7}$ | $\text{N·m}/(\text{rad/s})^2$ |
| $\omega_{\min}$ | 100 | rad/s |
| $\omega_{\max}$ | 838 | rad/s |

Hover RPM: $\omega_h = \sqrt{mg / (4 k_f)} \approx 515\,\text{rad/s}$

### 2.4 Motor Dynamics (First-Order Lag)

$$
\omega(t + \Delta t) = \omega(t) + \frac{\Delta t}{\tau_m} \cdot (\omega_{\text{cmd}} - \omega(t)), \quad \tau_m = 0.02\,\text{s}
$$

### 2.5 Mixer Matrix (Control Allocation)

$$
\begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix} = \mathbf{M} \begin{bmatrix} F_0 \\ F_1 \\ F_2 \\ F_3 \end{bmatrix}, \quad
\mathbf{M} = \begin{bmatrix}
1 & 1 & 1 & 1 \\
+d & +d & -d & -d \\
-d & +d & +d & -d \\
-0.016 & +0.016 & -0.016 & +0.016
\end{bmatrix}
$$

역할당: $\mathbf{F} = \mathbf{M}^{-1} \mathbf{w}$, 할당 후 $F_i < 0$이면 0으로 clamping.

---

## 3. Attitude Controller (Inner Loop, 300 Hz)

### 3.1 Thrust Computation

$$
\mathbf{g}^{(b)} = \mathbf{R}^T [0, 0, -g]^T, \quad
T = m \cdot (a_{\text{cmd},z}^{(b)} - g_z^{(b)}), \quad T \in [0.5mg,\; 4.0mg]
$$

### 3.2 Desired Attitude

가속도 명령으로부터 원하는 추력 방향(body Z축):

$$
\hat{\mathbf{z}}_{\text{des}} = \frac{\mathbf{R} \cdot \mathbf{a}_{\text{cmd}}^{(b)} + [0, 0, g]^T}{\|\cdot\|}
$$

Yaw reference $\psi_{\text{ref}}$와 cross-product로 $\mathbf{R}_{\text{des}} = [\hat{\mathbf{x}}_{\text{des}}, \hat{\mathbf{y}}_{\text{des}}, \hat{\mathbf{z}}_{\text{des}}]$ 구성.

### 3.3 SO(3) Attitude Error

$$
\mathbf{E}_R = \mathbf{R}_{\text{des}}^T \mathbf{R} - \mathbf{R}^T \mathbf{R}_{\text{des}}
$$

Vee map (skew-symmetric → vector):

$$
\mathbf{e}_R = \frac{1}{2} [E_{R,32} - E_{R,23},\; E_{R,13} - E_{R,31},\; E_{R,21} - E_{R,12}]^T
$$

Quaternion 대비 **antipodal ambiguity 없음**, large angle에서도 well-defined.

### 3.4 PD Torque Control

$$
\boldsymbol{\tau} = -\mathbf{K}_p \cdot \mathbf{e}_R + \mathbf{K}_d \cdot (\boldsymbol{\omega}_{\text{cmd}} - \boldsymbol{\omega})
$$

| 게인 | Roll | Pitch | Yaw |
|------|------|-------|-----|
| $K_p$ | 8.0 | 8.0 | 4.0 |
| $K_d$ | 1.5 | 1.5 | 0.8 |

---

## 4. Gripper Geometry & Constraints

그리퍼 중심: body 중심에서 $z$ 방향 $-0.08\,\text{m}$.

| 축 | 간격 | Box 크기 | 마진/side |
|----|------|---------|----------|
| X (strut) | $2 \times 0.05 = 0.10\,\text{m}$ | 0.08 m | **1 cm** |
| Y (plate tip) | $2 \times 0.104 = 0.208\,\text{m}$ | 0.08 m | **6.4 cm** |
| Y (strut level) | $2 \times 0.062 = 0.124\,\text{m}$ | 0.08 m | 2.2 cm |

그리퍼 속도 보정 (angular velocity offset):

$$
v_{gx} = v_{bx} - \omega_y \cdot 0.08, \quad v_{gy} = v_{by} + \omega_x \cdot 0.08
$$

---

## 5. PID Docking Controller (Outer Loop, 150 Hz)

**최종 성능**: 63.8% dock, 0% crash (dynamic box, DR 환경)

### 5.1 XY Control: Asymmetric PD with Gain Scheduling

$$
a_x = K_{p,x} \cdot e_x - K_{d,x} \cdot \dot{x}_w, \quad
a_y = K_{p,y} \cdot e_y - K_{d,y} \cdot \dot{y}_w
$$

**Dock proximity factor** $\delta$ (두 sigmoid의 곱):

$$
\delta = \sigma\!\left(\frac{0.04 - h}{0.015}\right) \cdot \sigma\!\left(\frac{0.03 - d_{xy}}{0.01}\right)
$$

$$
K_{p,x} = 12.0 - 2.0\,\delta, \quad K_{d,x} = 8.0 + 1.0\,\delta
$$

$$
K_{p,y} = 8.0 - 1.5\,\delta, \quad K_{d,y} = 6.0 + 1.0\,\delta
$$

| 상태 | $K_{p,x}$ | $K_{d,x}$ | $K_{p,y}$ | $K_{d,y}$ |
|------|-----------|-----------|-----------|-----------|
| 원거리 ($\delta=0$) | 12.0 | 8.0 | 8.0 | 6.0 |
| 도킹 ($\delta=1$) | 10.0 | 9.0 | 6.5 | 7.0 |

### 5.2 Z Control: PID with Two-Stage Gated Descent

XY 정렬 상태에 따라 목표 고도를 동적 설정:

$$
g_c = \sigma\!\left(\frac{0.10 - d_{xy}}{0.05}\right), \quad
g_f = \sigma\!\left(\frac{0.03 - d_{xy}}{0.01}\right)
$$

$$
h_{\text{target}} = 0.30 - 0.18 \cdot g_c - 0.10 \cdot g_f
$$

| $d_{xy}$ | $h_{\text{target}}$ | 의미 |
|----------|---------------------|------|
| > 15cm | 0.30m | 상공 대기 |
| ≈ 5cm | 0.13m | 안전 하강 |
| < 2cm | 0.02m | 도킹 진입 |

PID 제어 법칙 ($z_{\text{target}} = z_{\text{obj}} + h_{\text{target}}$):

$$
a_z = K_{p,z} \cdot e_z + K_{i,z} \cdot I_z - K_{d,z} \cdot \dot{z}_w
$$

| $K_{p,z}$ | $K_{i,z}$ | $K_{d,z}$ |
|-----------|-----------|-----------|
| 12.0 | 3.0 | 7.0 |

**Integral anti-windup**:

$$
I_z(t+1) = \begin{cases}
\text{clamp}(I_z + e_z \Delta t,\; -1.0,\; 1.0) & \text{integrating} \\
0.95 \cdot I_z & \text{otherwise (leak)}
\end{cases}
$$

과거 PID 실패(integral overshoot)를 gated descent + clamp + leak으로 해결.

### 5.3 Dock Hold & Climb

| `contain_hold_count` | 상태 | Z 목표 | 그리퍼 |
|--------------------|------|--------|--------|
| 0–149 | Approach | $z_{\text{obj}} + h_{\text{target}}$ | Open (0.873 rad) |
| 150–324 | Closing | $z_{\text{obj}} + 0.04$ (고정) | Closing |
| ≥ 325 | Grasped | Climb to 1.5m | Closed (-0.087 rad) |

Climb mode: $K_p^{\text{climb}} = 6.0$, $K_d^{\text{climb}} = 5.0$ (저게인으로 box 이탈 방지).

### 5.4 Stuck Detection & Recovery

Detection (4조건, 100 step 지속):

$$
(h \in (-0.05, 0.06)) \wedge (d_{xy} > 0.04) \wedge (\|\mathbf{v}\| < 0.12) \wedge (\text{contain} < 10)
$$

Recovery (150 step): $z_{\text{obj}} + 0.30\,\text{m}$로 pull-up, XY 추가 보정 ($K_p$=4.0).

Safety: $h < -0.02\,\text{m}$이면 $a_z \mathrel{+}= 3.0$. 모든 축 clamp $\pm 8.0\,\text{m/s}^2$.

---

## 6. Dock Detection: Overlap-Based Containment

### 6.1 Overlap Ratio

Strut level에서 2D AABB intersection:

$$
\text{overlap\_ratio} = \frac{\text{overlap}_x \cdot \text{overlap}_y}{(0.08)^2}
$$

- Strut X: $\pm 0.05\,\text{m}$, Strut Y: $\pm 0.062\,\text{m}$
- Z gate: $z_{\text{box}}^{(l)} \in [-0.12, 0.02]$
- **Full contain**: overlap_ratio > 0.50

### 6.2 Dock Success

- `contain_hold_count` ≥ 225 (1.5초 유지) → Stage 3 학습 성공 기준
- `contain_hold_count` ≥ 325 (2.17초) → End-to-end mission 기준

---

## 7. Observation & Action Spaces

### 7.1 31D Observation (GripperDroneEnv)

$$
\mathbf{o} = [\mathbf{v}_g^{(b)},\; \boldsymbol{\omega}^{(b)},\; \text{vec}(\mathbf{R}),\; \mathbf{p}_{\text{goal}}^{(g)},\; \mathbf{e}_{\text{goal}}^{(w)},\; \hat{\mathbf{a}}_{xy},\; \hat{\boldsymbol{\theta}},\; \mathbf{p}_{\text{obj}}^{(g)},\; f_{\text{grasp}},\; \hat{m}_p,\; d_{\text{obj}}]
$$

| 성분 | 차원 | 정의 |
|------|------|------|
| $\mathbf{v}_g^{(b)}$ | 3 | Gripper 속도 (각속도 보정) |
| $\boldsymbol{\omega}^{(b)}$ | 3 | 각속도 |
| $\text{vec}(\mathbf{R})$ | 9 | 회전행렬 (singularity-free) |
| $\mathbf{p}_{\text{goal}}^{(g)}$ | 3 | Goal in gripper frame |
| $\mathbf{e}_{\text{goal}}^{(w)}$ | 3 | Goal error in world frame |
| $\hat{\mathbf{a}}_{xy}$ | 2 | 이전 XY action |
| $\hat{\boldsymbol{\theta}}$ | 2 | 정규화 plate 각도 |
| $\mathbf{p}_{\text{obj}}^{(g)}$ | 3 | Object in gripper frame |
| $f_{\text{grasp}},\hat{m}_p,d_{\text{obj}}$ | 3 | Grasp flag, payload, distance |

### 7.2 8D Action

| 인덱스 | 물리량 | 범위 |
|--------|--------|------|
| 0-2 | Body accel | $\pm 8.0\,\text{m/s}^2$ |
| 3-5 | Body rates | $\pm 3.0\,\text{rad/s}$ |
| 6 | Yaw ref | $\pm \pi$ |
| 7 | Gripper angle | $[-0.087, 0.873]\,\text{rad}$ |

Stage 3 hybrid blending: $\mathbf{a}_{\text{final}} = \mathbf{a}_{\text{PID}} + s \cdot \mathbf{a}_{\text{RL}}$ ($s = 0.0$ in end-to-end).

---

## 8. Domain Randomization

| 파라미터 | 범위 | 물리적 의미 |
|---------|------|----------|
| Mass scale | $U(0.9, 1.1)$ | 총 질량 ±10% |
| Motor $k_f$ scale | $U(0.85, 1.15)$ | 추력 효율 ±15% |
| Wind force | $0.5\,\text{N}$, $0.5\,\text{Hz}$ | $\mathbf{F}_{\text{wind}} = A \sin(2\pi f t + \boldsymbol{\phi})$ |
| Position noise | $\sigma = 0.02\,\text{m}$ | 위치 센서 |
| Velocity noise | $\sigma = 0.05\,\text{m/s}$ | 속도 센서 |
| Object detection noise | $\sigma = 0.03\,\text{m}$ | 물체 인식 |
| Payload mass (S4) | $U(0.15, 0.25)\,\text{kg}$ | 적재 변동 |

---

## 9. Simulation Environment (Isaac Lab)

| 파라미터 | 값 |
|---------|-----|
| Simulation dt | $1/300\,\text{s}$ (300 Hz) |
| Gravity | $-9.81\,\text{m/s}^2$ |
| Num envs | 4096 |
| Grasp object | 8×8×8 cm, 0.2 kg, dynamic |
| Pedestal | 30×30×50 cm, kinematic |
| Gripper actuator | stiffness=300, damping=15, effort=50 N |

## 10. Tuning History & Failure Modes

### 10.1 PID Tuning Progression

| 단계 | 변경 | Dock% |
|------|------|-------|
| 1. Initial PD (Kp=6) | baseline | 24.7% |
| 6. Asymmetric gains | Kp_x=12, Kp_y=6 | 49.2% |
| 9. Stuck retry | pull-up recovery | 57.9% |
| 10. Eval fix + PID | integral term | **63.8%** |

### 10.2 Failure Breakdown (63.8%)

| 결과 | 비율 | 원인 |
|------|------|------|
| DOCK (성공) | 63.8% | — |
| FAR | 21.1% | XY 접근 실패 |
| NO_DESCENT | 7.9% | 순간 정렬만 |
| OVERLAP_LOST | 7.2% | 접촉 시 box 밀림 |

### 10.3 INDI / 과거 PID 실패 요약

- **INDI**: outer loop에서 inner loop 응답 지연과 충돌 → 발산 (dock 0.1%)
- **과거 PID**: approach 중 integral 과누적 → overshoot. 현재 gated descent + leak으로 해결

데이터: `data_csv/pd_tuning_progression.csv`, `data_csv/dock_vs_overlap_lost_at_first_overlap.csv`
