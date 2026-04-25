# Waypoint Navigation RL: Stage 1 (Approach) & Stage 4 (Loaded Delivery)

## 1. Overview

End-to-end 미션에서 장거리 이동을 담당하는 RL 정책.
Analytical controller가 precision docking에 적합한 반면, **waypoint navigation**과 **적재 비행 적응**에는
RL이 더 효과적이다. 동일한 PPO 프레임워크와 네트워크 구조를 공유하되,
Stage 4는 payload mass 정보가 추가된 23D 관측 공간을 사용한다.

| 구분 | Stage 1 (Flight) | Stage 4 (Loaded) |
|------|-----------------|-----------------|
| 역할 | 비적재 접근/복귀 비행 | 적재 상태 배달 비행 |
| Obs dim | 22D | 23D (+payload mass) |
| Action dim | 4D | 4D |
| Tilt 제한 | 0.52 rad (30°) | 0.35 rad (20°) |
| 최종 모델 | `gripper_wp_flight_v6` | `gripper_wp_loaded_v23` (학습 중) |

---

## 2. 관측 공간 (Observation Space)

### 2.1 Stage 1: 22D Observation

모든 관측값은 **body frame** 기준으로 구성되며, sensor noise가 적용된다:

$$
\mathbf{o} = [\underbrace{\mathbf{v}_b}_{3}, \underbrace{\boldsymbol{\omega}_b}_{3}, \underbrace{\text{vec}(\mathbf{R})}_{9}, \underbrace{\mathbf{g}_b}_{3}, \underbrace{\hat{\mathbf{a}}_{t-1}}_{4}] \in \mathbb{R}^{22}
$$

| 성분 | 차원 | 정의 | Noise |
|------|------|------|-------|
| $\mathbf{v}_b$ | 3 | Body-frame 선속도 | $\mathcal{N}(0, 0.03^2)$ |
| $\boldsymbol{\omega}_b$ | 3 | Body-frame 각속도 | — |
| $\text{vec}(\mathbf{R})$ | 9 | $3\times3$ 회전행렬 평탄화 | — |
| $\mathbf{g}_b$ | 3 | Goal error in body frame: $\mathbf{R}^T(\mathbf{p}_{\text{goal}} - \mathbf{p}_{\text{drone}})$ | $\mathcal{N}(0, 0.01^2)$ |
| $\hat{\mathbf{a}}_{t-1}$ | 4 | 이전 action 정규화: $[a_{xyz}/8,\; a_{\text{yaw}}/\pi]$ | — |

**회전행렬 사용 근거**: Euler angle은 gimbal lock, quaternion은 antipodal ambiguity 문제가 있다.
$3\times3$ 행렬 표현은 9D이지만 singularity-free이며 RL 학습에 가장 안정적이다.

### 2.2 Stage 4: 23D Observation

Stage 1의 22D에 payload mass 추정값 1D를 추가:

$$
\mathbf{o}_{\text{loaded}} = [\mathbf{o}_{22},\; \hat{m}_{\text{payload}}] \in \mathbb{R}^{23}
$$

$$
\hat{m}_{\text{payload}} = m_{\text{payload}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.02^2)
$$

- 실제 payload mass: $m \sim U(0.15, 0.25)\,\text{kg}$ (학습 시)
- Noise 0.02 kg: 센서 불확실성 모사

### 2.3 Observation Preprocessing

SKRL의 `RunningStandardScaler`로 online normalization:

$$
\hat{o}_i = \frac{o_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad \epsilon = 10^{-8}
$$

$\mu$와 $\sigma^2$는 학습 중 exponential moving average로 갱신.

---

## 3. 행동 공간 (Action Space)

### 3.1 4D Action

정책 네트워크 출력 $\mathbf{a} \in [-1, 1]^4$을 물리 단위로 스케일링:

$$
\mathbf{u} = \mathbf{a}_{\text{low}} + \frac{\mathbf{a} + 1}{2} \cdot (\mathbf{a}_{\text{high}} - \mathbf{a}_{\text{low}})
$$

| 인덱스 | 물리량 | 범위 | 단위 |
|--------|--------|------|------|
| 0-2 | Body-frame 가속도 $[a_x, a_y, a_z]$ | $[-8.0, 8.0]$ | $\text{m/s}^2$ |
| 3 | Yaw reference $\psi_{\text{ref}}$ | $[-\pi, \pi]$ | rad |

### 3.2 8D 환경 매핑

4D waypoint action → 8D environment action으로 매핑:

```
action_8d[0:3] = action_4d[0:3]   # accel xyz
action_8d[3:6] = 0                 # rate commands (unused)
action_8d[6]   = action_4d[3]     # yaw reference
action_8d[7]   = 0                 # gripper (auto-controlled)
```

---

## 4. 보상 함수 (Reward Functions)

### 4.1 Stage 1: Waypoint Flight Reward

**GripperWaypointEnv 내 보상** (`gripper_waypoint_env.py`):

#### r_direction — 목표 방향 속도 보상

$$
r_{\text{dir}} = 0.5 \cdot \text{clamp}\!\left(\frac{\mathbf{v}_w \cdot \hat{\mathbf{d}}}{1.0},\; -3.0,\; 3.0\right)
$$

여기서 $\hat{\mathbf{d}} = (\mathbf{p}_{\text{goal}} - \mathbf{p}_{\text{drone}}) / \|\cdot\|$. 목표를 향해 이동하면 양의 보상.

#### r_arrive — 도달 보너스 (시간 감쇠)

$$
r_{\text{arrive}} = \begin{cases}
\displaystyle \frac{10.0}{t_{\text{goal}} / 150 + 0.5} & \text{if } d_{\text{goal}} < 0.3\,\text{m} \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

$t_{\text{goal}}$: 현재 waypoint에 도달 후 경과 step 수.
**빠른 도달에 높은 보상**: $t=0$이면 $r=20$, $t=150$ (1초)이면 $r=6.67$.

#### r_crash — 추락 패널티

$$
r_{\text{crash}} = -5.0 \cdot \mathbb{1}[z_{\text{local}} < 0.15\,\text{m}]
$$

#### r_smooth — 행동 부드러움

$$
r_{\text{smooth}} = -0.01 \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2
$$

#### r_angular — 각속도 억제

$$
r_{\text{angular}} = -0.02 \cdot \|\boldsymbol{\omega}_b\|
$$

#### r_tilt — 기울기 패널티

$$
r_{\text{tilt}} = -3.0 \cdot \max(0,\; \theta_{\text{tilt}} - 0.52)
$$

여기서 $\theta_{\text{tilt}} = \arccos(\mathbf{R}_{33})$, 임계값 0.52 rad ≈ 30°.

#### r_timeout — Waypoint 시간 초과

$$
r_{\text{timeout}} = -2.0 \cdot \mathbb{1}[t_{\text{wp}} > 450]
$$

450 step = 3초. 한 waypoint에 3초 이상 걸리면 패널티 + 다음 WP로 전환.

#### 총 보상

$$
R_{\text{S1}} = r_{\text{dir}} + r_{\text{arrive}} + r_{\text{crash}} + r_{\text{smooth}} + r_{\text{angular}} + r_{\text{tilt}} + r_{\text{timeout}}
$$

### 4.2 Stage 4: Loaded Flight Reward

Stage 1과 동일한 구조이되, **적재 안정성**을 위해 tilt 제한이 강화된다:

$$
r_{\text{tilt}}^{\text{loaded}} = -4.0 \cdot \max(0,\; \theta_{\text{tilt}} - 0.35)
$$

| 항목 | Stage 1 | Stage 4 |
|------|---------|---------|
| Tilt 임계값 | 0.52 rad (30°) | 0.35 rad (20°) |
| Tilt 패널티 가중치 | -3.0 | -4.0 |
| 물리적 근거 | 빈 드론, 기동성 허용 | 적재 시 급기동 → box 이탈 위험 |

나머지 보상 항목은 Stage 1과 동일.

### 4.3 Stage 1 (reward_fn.py 내): Basic Flight Reward

GripperDroneEnv의 Stage 1 학습 시 사용되는 보상 (waypoint env가 아닌 원래 환경):

$$
r_{\text{pos}} = 4.0 \cdot \exp(-1.2 \cdot d_{\text{goal}})
$$

$$
r_{\text{arrive}} = 10.0 \cdot \mathbb{1}[d_{\text{goal}} < 0.5\,\text{m}]
$$

$$
r_{\text{hover}} = 5.0 \cdot \mathbb{1}[\text{arrived}] \cdot \exp(-3.0 \cdot \|\mathbf{v}\|)
$$

$$
r_{\text{smooth}} = -0.1 \cdot \|\Delta\mathbf{a}\|^2, \quad r_{\text{mag}} = -0.02 \cdot \|\mathbf{a}\|^2
$$

### 4.4 Stage 4 (reward_fn.py 내): Loaded Flight Reward

**Shifted exponential** 설계 — 모든 position reward가 $\leq 0$이어서 hovering exploit 방지:

$$
r_{\text{time}} = -0.5 \quad \text{(매 step)}
$$

$$
r_{\text{pos}} = 0.5 \cdot (\exp(-1.2 \cdot d_{\text{goal}}) - 1.0) \leq 0
$$

$$
r_{\text{level}} = 0.6 \cdot \exp(-2.0 \cdot \theta_{\text{tilt}})
$$

$$
r_{\text{smooth}} = -0.1 \cdot \|\Delta\mathbf{a}\|^2, \quad r_{\text{mag}} = -0.02 \cdot \|\mathbf{a}\|^2
$$

$$
r_{\text{arrive}} = 200.0 \quad \text{(도달 시 1회 지급)}
$$

**보상 순서**: arrive ($+200$) ≫ survive ($\geq 0.1$/step) > crash ($=0$) > hover-forever ($< 0$/step).
이 구조는 **도달하지 않으면 시간이 갈수록 총 보상이 감소**하도록 설계되어,
hovering exploit (이동하지 않고 intermediate reward 수집)을 원천 차단한다.

---

## 5. 네트워크 구조

### 5.1 Policy Network

$$
\boldsymbol{\mu} = f_\theta(\mathbf{o}), \quad \mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu},\; \text{diag}(\exp(\log\boldsymbol{\sigma})^2))
$$

```
Input (22D or 23D)
  → Linear(obs_dim, 128) + ELU
  → Linear(128, 128) + ELU
  → Linear(128, 4) → mean μ
  + Learnable log_σ (1D, shared across actions)
```

- Weight init: Orthogonal (gain = 0.01) — 초기 action 분산을 작게 유지
- $\log\sigma$ 범위: $[-2.0, 2.0]$ → $\sigma \in [0.135, 7.39]$
- 총 파라미터: ~22k

### 5.2 Value Network

$$
V_\phi(\mathbf{o}) = g_\phi(\mathbf{o}) \in \mathbb{R}
$$

```
Input (22D or 23D)
  → Linear(obs_dim, 128) + ELU
  → Linear(128, 128) + ELU
  → Linear(128, 1) → value estimate
```

- Weight init: Orthogonal (gain = 1.0)
- 총 파라미터: ~16.7k

### 5.3 Transfer Learning (Stage 1 → Stage 4)

Stage 4 모델은 Stage 1의 converged weights로 warm-start:

- 첫 번째 layer: 22 columns 복사, 23번째 column (payload input) → zero init
- 나머지 layers: 직접 복사 (동일 크기)
- 효과: waypoint navigation 능력을 유지하면서 payload 적응만 학습

---

## 6. PPO 학습 설정

### 6.1 Hyperparameters

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Rollout length | 100 steps | 환경 당 수집 step |
| Learning epochs | 5 | 데이터 재사용 횟수 |
| Mini-batches | 4 | 배치 분할 수 |
| Discount $\gamma$ | 0.99 | 보상 감쇄 |
| GAE $\lambda$ | 0.95 | Advantage 추정 |
| Learning rate | $3 \times 10^{-4}$ | Adam optimizer |
| Clip ratio $\epsilon$ | 0.2 | PPO surrogate clip |
| Value clip | 0.2 | Value function clip |
| Entropy coeff | 0.004 | 탐색 유지 |
| Value loss coeff | 1.0 | Value loss 가중치 |
| Grad norm clip | 1.0 | Gradient clipping |
| Num envs | 4096 | 병렬 환경 수 |

### 6.2 PPO Objective

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
$$

**GAE advantage estimation**:

$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### 6.3 학습 환경 설정

| 파라미터 | 값 |
|---------|-----|
| Episode length | 20.0 s |
| Simulation dt | 1/300 s (300 Hz) |
| Decimation | 2 (policy at 150 Hz) |
| WP 개수 | 4 per episode |
| WP 간 거리 | 0.8 - 2.3 m (random) |
| WP timeout | 450 steps (3.0 s) |
| Z range (flight) | [1.0, 3.5] m |
| Z range (loaded) | [0.8, 3.0] m |
| XY bounds | ±4.0 m from origin |

---

## 7. 종료 조건 (Termination)

| 조건 | 임계값 | 종류 |
|------|--------|------|
| Too low | $z_{\text{local}} < 0.15\,\text{m}$ | 추락 |
| Too far | $d_{xy} > 10.0\,\text{m}$ | 이탈 |
| Too tilted | $\theta > 60°$ | 불안정 |
| Box dropped | $z_{\text{box,local}} < 0.30\,\text{m}$ | Stage 4 only |
| WP timeout | 450 steps (3s) | Truncation (reset WP) |

---

## 8. Domain Randomization

| 파라미터 | 범위 | 영향 |
|---------|------|------|
| Mass scale | $U(0.9, 1.1)$ | 총 질량 ±10% |
| Motor $k_f$ scale | $U(0.85, 1.15)$ | 추력 계수 ±15% |
| Wind | $0.5\,\text{N}$, $f = 0.5\,\text{Hz}$ | 시변 외란 |
| Payload mass (S4) | $U(0.15, 0.25)\,\text{kg}$ | 적재 변동 |

---

## 9. 현재 모델 상태

### Stage 1: `gripper_wp_flight_v6`
- **상태**: 수렴 완료
- Checkpoint: `final_agent.pt` (485 KB)
- 학습: ~2B total steps (4096 envs)
- 성능: 94% waypoint 도달률, 정밀 도달(<0.3m) 86%

### Stage 4: `gripper_wp_loaded_v23`
- **상태**: 학습 중 (2026-04-25 현재)
- Transfer: `gripper_wp_flight_v6`에서 warm-start
- End-to-end eval에서는 `v20/best_agent.pt` 사용 중

---

## 10. 데이터 파일

- `../data_csv/waypoint_eval_flight_*.json`: 비행 eval 결과 (5개)
- `../data_csv/learning_curve_*.csv`: 학습 곡선 (TensorBoard export)
