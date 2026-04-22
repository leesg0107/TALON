# Stage 1/4 Redesign Plan: Simple Waypoint Follower

## 연구 분석: 성공적인 RL 드론 제어

### 공통 패턴 (Kaufmann 2023, Song 2023, Loquercio 2021, Sun 2022)

**Action space**: `[collective_thrust, body_rate_x, body_rate_y, body_rate_z]` = 4D
- 직접 motor command가 아닌 **thrust + body rates**가 표준
- Attitude controller가 이를 motor 명령으로 변환

**Observation space**: 12-16D, body-centric
- Body velocity (3D)
- Angular velocity (3D)
- Gravity vector in body frame (3D) — R 대신 (9D→3D 절약)
- Goal relative position in body frame (3D)
- (선택) Previous action (3-4D)

**Reward**: 극도로 단순
- Distance to goal (exp decay)
- Arrival bonus
- Hover bonus (도달 후 정지)
- Action smoothness penalty

**Training**: PPO, 4096+ parallel envs, domain randomization

### 핵심 인사이트

1. **Gravity vector vs Rotation matrix**: `g_b = R^T @ [0,0,-g]`의 방향 벡터(3D)가 전체 R(9D)보다 효율적. Tilt 정보를 직접 제공. Yaw는 goal_b에 이미 반영됨.

2. **Action = thrust + rates (4D)가 표준**: 우리 시스템은 `accel_b(3) + rate_b(3) + yaw(1) + gripper(1) = 8D`. 단순화: `accel_b(3) + yaw(1) = 4D`, rate_b=0 고정.

3. **Attitude controller가 핵심 안정 장치**: 우리 AttitudeController가 accel_cmd에서 desired attitude를 자동 계산. RL은 "어디로 가고 싶은지"만 지시, 자세 안정화는 controller가 담당.

## 설계안

### Architecture

```
SimpleDroneEnv (Stage 1/4 공용)
├── Obs: 15D (vel_b:3, ang_vel_b:3, gravity_b:3, goal_b:3, prev_accel:3)
├── Action: 4D (accel_b:3, yaw_ref:1)
├── Reward: distance + arrive + hover + smooth
└── Uses: 기존 AttitudeController + MotorModel (변경 없음)
```

### Observation (15D)

| Dim | Name | Range | 설명 |
|-----|------|-------|------|
| 0-2 | vel_b | ±5 m/s | Body-frame linear velocity |
| 3-5 | ang_vel_b | ±3 rad/s | Body-frame angular velocity |
| 6-8 | gravity_b | ±1 | Gravity direction in body frame (= -R[:,2]) |
| 9-11 | goal_b | ±5 m | Goal position relative to drone, body frame |
| 12-14 | prev_accel | ±1 | Previous action (normalized) |

**Stage 4 추가**: `payload_mass` (1D) → 16D total

**gravity_b 계산**:
```python
gravity_b = -R[:, :, 2]  # = R^T @ [0,0,-1] 의 방향 (3D)
# R[:,0,2], R[:,1,2], R[:,2,2] → 드론이 기울어진 정도를 직접 표현
```

### Action (4D)

| Dim | Name | Range | 설명 |
|-----|------|-------|------|
| 0-2 | accel_cmd_b | [-8, 8] m/s² | Body-frame desired acceleration |
| 3 | yaw_ref | [-π, π] | Desired yaw angle |

**rate_cmd_b = [0, 0, 0]** 고정 → Attitude controller가 accel에서 자동 계산.
**gripper = 0** 고정 (Stage 1) 또는 **-1** 고정 (Stage 4, 잡은 상태).

### 기존 AttitudeController 호환

```python
# SimpleDroneEnv._apply_action():
accel_cmd_b = scaled_actions[:, :3]          # RL output
rate_cmd_b = torch.zeros(N, 3, device=dev)    # 고정 0
yaw_ref = scaled_actions[:, 3]                # RL output

forces, torques = self.attitude_ctrl.compute(
    accel_cmd_b, rate_cmd_b, yaw_ref, quat_w, ang_vel_b, dt
)
```

**AttitudeController는 변경 없음** — 동일한 compute() 인터페이스.

### Reward

```python
@dataclass
class WaypointRewardWeights:
    w_pos: float = 5.0          # position tracking
    a_pos: float = 1.5          # exp sharpness
    w_arrive: float = 15.0      # arrival bonus
    arrive_threshold: float = 0.3  # m
    w_hover: float = 8.0        # hover at goal
    a_hover_vel: float = 3.0    # exp sharpness
    w_smooth: float = 0.1       # action smoothness
    w_mag: float = 0.02         # action magnitude

def compute_waypoint_reward(pos_err, vel_mag, action, prev_action):
    r_pos = 5.0 * exp(-1.5 * pos_err)
    r_arrive = 15.0 * (pos_err < 0.3)
    r_hover = 8.0 * (pos_err < 0.3) * exp(-3.0 * vel_mag)
    r_smooth = -0.1 * ||action - prev_action||²
    r_mag = -0.02 * ||action||²
    return r_pos + r_arrive + r_hover + r_smooth + r_mag
```

### Training Config

```python
# PPO (SKRL)
num_envs = 4096
rollouts = 24
learning_rate = 3e-4
discount_factor = 0.99
lambda_ = 0.95
entropy_loss_scale = 0.005
mini_batches = 24

# Network: [256, 128, 64] (연구들의 표준)
# 더 작은 네트워크 — 15D obs에 512→256→128은 과대

# Domain randomization
mass_scale = (0.9, 1.1)
motor_kf_scale = (0.85, 1.15)
wind_force_std = 0.5  # N
```

### Environment Config

**Stage 1 (Approach flight)**:
```python
episode_length_s = 6.0
spawn_z = 3.0  # ±0.5m
goal_range_xy = 2.0  # ±2m
goal_range_z = (1.5, 3.5)
max_tilt_deg = 60
min_altitude = 0.3
```

**Stage 4 (Loaded flight)**:
```python
episode_length_s = 6.0
spawn_z = 2.0  # 파지 후 고도
goal_range_xy = 2.0
goal_range_z = (1.5, 3.0)
payload_mass_range = (0.15, 0.25)  # box 0.2kg ± variation
```

### Phase 전환 안정성

```
Stage 1 (SimpleDroneEnv) → PD dock (GripperDroneEnv) → Stage 4 (SimpleDroneEnv)
```

**전환 시 공유되는 것**: AttitudeController (동일한 compute 인터페이스)
**전환 시 바뀌는 것**: obs 계산, action 해석, reward

**안정성 보장**:
- Stage 1 마지막 상태 (position, velocity, attitude) → PD가 그대로 이어받음
- PD는 raw physics state를 직접 읽으므로 obs 형식 무관
- PD dock 완료 후 → Stage 4가 이어받음 (attitude controller mass 업데이트)

**End-to-end demo에서의 action 매핑**:
```python
# Stage 1/4의 4D output → GripperDroneEnv의 8D action
action_8d = torch.zeros(1, 8, device=dev)
action_8d[:, :3] = simple_policy_output[:, :3]  # accel
action_8d[:, 6] = simple_policy_output[:, 3]    # yaw
# [3:6] = 0 (rate_cmd), [7] = 0 (gripper)
```

### 파일 구조

```
envs/
├── drone_env.py          # 기존 (Stage 3 PD + grasping)
├── env_cfg.py            # 기존 config
├── waypoint_env.py       # NEW: 15D/4D simple waypoint env
└── waypoint_cfg.py       # NEW: waypoint env config

rewards/
├── reward_fn.py          # 기존 (Stage 1-5)
└── waypoint_reward.py    # NEW: 단순 waypoint reward

train_waypoint.py          # NEW: Stage 1/4 학습 스크립트
scripts/
├── eval_waypoint.py       # NEW: Stage 1/4 평가
└── eval_end_to_end.py     # 수정: 새 모델 사용
```

### 학습 순서

1. **Stage 1 학습** (~2-3시간)
   ```bash
   python train_waypoint.py --stage 1 --num_envs 4096 --max_steps 500000000
   ```

2. **Stage 1 평가** (도달 정확도 <30cm 95%+ 목표)
   ```bash
   python scripts/eval_waypoint.py --stage 1
   ```

3. **Stage 4 학습** (~2-3시간, payload 추가)
   ```bash
   python train_waypoint.py --stage 4 --num_envs 4096 --max_steps 500000000
   ```

4. **End-to-end 통합 테스트**
   ```bash
   python scripts/eval_end_to_end.py
   ```
