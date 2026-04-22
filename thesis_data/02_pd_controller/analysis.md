# PD Analytical Controller Analysis

## Overview

Pure RL의 best(24.8% dock)을 대체하기 위해 analytical PD controller를 설계.
최종 성능: **63.8% dock, 0% crash** (eval 버그 수정 후).
RL 없이, 학습 시간 0으로 달성.

## Controller Architecture

```
Input: drone state (pos, vel, quat) + box position
Output: body-frame acceleration [ax, ay, az] → attitude controller → motors

XY: World-frame PD with asymmetric gains (yaw≈0 가정)
Z:  Gated descent (XY 정렬 시에만 하강)
```

### XY Control

```python
Kp_x = 12.0, Kd_x = 8.0   # X: strut direction (margin 1cm) — tight
Kp_y = 6.0,  Kd_y = 5.0    # Y: plate direction (margin 6.4cm) — relaxed
```

**비대칭 게인 근거**: X축 마진 1cm vs Y축 마진 6.4cm. X에 2배 높은 Kp로 정밀 추적.

**Dock-aware gain scheduling**:
```python
dock_proximity = sigmoid((0.10 - alt_above) / 0.03) * sigmoid((0.05 - xy_mag) / 0.02)
Kp_x = 12.0 → 6.0 (near box)    # 접촉 시 부드럽게
Kd_x = 8.0 → 12.0 (near box)    # 감쇠 강화
```

### Z Control (Gated Descent)

```python
descent_gate = sigmoid((0.10 - xy_mag) / 0.03)  # center=10cm
```
XY가 10cm 이내일 때만 하강. 정렬 안 된 상태에서 하강하면 box edge 충돌.

**Adaptive speed**:
```python
speed_scale = sigmoid((alt_above - 0.15) / 0.05)
max_vz = -0.15 (near box) ~ -0.40 (high altitude)
```
높을 때 빠르게, box 근처에서 느리게 → soft contact.

**Z damping**:
```python
Kp_z = 4.0 (far) → 8.0 (near box)
```
Box 근처에서 감쇠 2배 → 빠른 감속.

### Stuck Detection + Pull-up Retry

렌더링 관찰에서 발견: 한쪽 strut이 box edge에 걸려 드론이 freeze.

```python
stuck_condition = (alt < 12cm) & (xy > 1.5cm) & (vel < 15cm/s) & (descent_gate > 0.3)
40 steps (0.27s) 지속 → pull up at 0.4 m/s
```

**효과**: dock 52.7% → 57.9% (+5.2%). 단일 변경 중 가장 큰 개선.

## Tuning 과정 (data_csv/pd_tuning_progression.csv)

| 단계 | 핵심 변경 | Dock% | 핵심 발견 |
|------|----------|-------|----------|
| 1. 초기 PD (Kp=6,Kd=4.5) | baseline | 24.7% | PD만으로 RL(24.8%)과 동등 |
| 2. Kp=8, Kd=5 | P gain 증가 | 32.2% | XY 수렴 가속 |
| 3. Kp=8, Kd=6.5 | D gain 증가 | 39.6% | oscillation 감쇄 |
| 4. Kp=8, Kd=7 | D 추가 증가 | 39.8% | sweet spot |
| 5. +adaptive descent, 12s | 하강 감속+시간 | 46.8% | OVERLAP_LOST 감소 |
| 6. 비대칭 게인 | Kp_x=12, Kp_y=6 | 49.2% | X 정밀도 향상 |
| 7. gate 10cm | 날카로운 gate | 52.7% | 정렬 후 하강 강제 |
| 8. Z damping | Kp_z 4→8 near | 51.1% | 접촉 속도 감소 |
| 9. stuck retry | pull-up on edge | 57.9% | frozen episode 구제 |
| 10. eval 수정 | reset bug fix | **63.8%** | 진짜 성능 확인 |

## PID/INDI 시도 및 실패

### PID (Integral term 추가)

| 설정 | Dock% | 변화 | 원인 |
|------|-------|------|------|
| Ki=2.0, clamp ±0.3 | 55.8% | -2.1% | approach 중 integral 누적 → 도착 시 overshoot |
| Ki=1.0, clamp ±0.2 | 53.2% | -4.7% | 같은 overshoot 문제 |
| Ki=1.5, conditional (xy<20cm) | 54.7% | -3.2% | 조건부 적용도 효과 없음 |

**PID 실패 원인**: approach 중 큰 error가 integral을 한 방향으로 누적 → box 근처 도착 시 integral이 overshoot 유발. FAR의 원인이 steady-state error가 아니라 transient dynamics여서 integral이 무의미.

### INDI (Incremental NDI)

| 설정 | Dock% | 문제 |
|------|-------|------|
| α=0.3, gain=1.0 | 6.0% | 완전 발산: v_z>0 (상승), tilt 18-25° |
| α=0.1, gain=0.3 | 0.1% | 여전히 발산 |

**INDI 실패 원인**: Outer loop INDI가 inner loop (attitude controller)와 충돌. 
Attitude controller의 응답 지연 (motor lag τ=0.02s) 때문에 INDI가 과도 보정 → 양성 피드백 → 발산.
INDI는 actuator에 가장 가까운 inner loop에서만 유효.

## 왜 PD인가 (PID, INDI가 아닌)

1. **Integral 불필요**: 실패 원인이 steady-state error가 아님. Motor kf ±15%, mass ±10%의 영향은 있지만, FAR 원인의 주류가 아님 (transient dynamics, wind, spawn 거리).

2. **INDI 불가**: Outer position loop에서 INDI는 inner attitude loop의 지연과 충돌. 구조적으로 안 맞음.

3. **PD의 강점**: 
   - Asymmetric gains로 gripper 기하학 반영
   - Dock-aware scheduling으로 접근/안착 모드 부드러운 전환
   - Stuck detection으로 edge-contact freeze 해결
   - 학습 시간 0, 디버깅 투명

## Failure Mode Analysis (63.8% 시점)

### Failure Breakdown
```
DOCK:          63.8% (754/1181)
OVERLAP_LOST:   7.2% (85/1181)  — contact 시 box 밀림
NO_DESCENT:     7.9% (93/1181)  — 순간 XY 정렬만
FAR:           21.1% (249/1181) — XY 접근 실패
```

### DOCK vs OVERLAP_LOST (first overlap 비교)
| Metric | DOCK | OVERLAP_LOST |
|--------|------|-------------|
| XY err | 4.4cm | 5.2cm |
| v_z | -0.325 m/s | -0.370 m/s |
| |x| < 2cm | 37% | 25% |

OVERLAP_LOST는 접촉 시 XY 정밀도가 낮고 하강 속도가 빠름.

### Spawn Distance vs Dock Rate
| Spawn range | Dock% | Far% |
|-------------|-------|------|
| 0-50cm | 68.5% | 15.7% |
| 50cm-1m | 66.6% | 17.8% |
| 1-2m | 36.4% | 56.8% |
| >2m | 0% | ~95% |

50cm 이내에서도 15.7% FAR — PD 수렴 한계 (DR: wind, motor kf).

### 렌더링 관찰 (핵심)
실패하는 케이스의 대부분: **한쪽 그리퍼 strut이 box edge에 먼저 닿아서 drone이 freeze.** 
다시 상승해서 재시도하지 않고 그 상태로 episode 종료.
→ Stuck detection + pull-up retry로 57.9→63.8% 개선.

## Domain Randomization (기본 활성화)

현재 eval/training 환경에 이미 DR 적용:
```
mass_scale:     0.9 ~ 1.1 (±10%)
motor_kf:       0.85 ~ 1.15 (±15%)
wind_force_std: 0.5N (~0.46 m/s² 교란)
pos_noise:      0.02m
vel_noise:      0.05 m/s
obj_detection_noise: 0.03m
detection_delay: 3 frames
```

PD controller는 clean state를 직접 읽지만, wind와 motor/mass variation은 물리적으로 영향.
같은 PD인데 episode마다 결과가 다른 이유: motor kf ±15%가 가장 큰 요인.

## 데이터 파일
- `../data_csv/pd_tuning_progression.csv`: 게인 튜닝 전체 과정
- `../data_csv/dock_vs_overlap_lost_at_first_overlap.csv`: 성공/실패 접촉 비교
