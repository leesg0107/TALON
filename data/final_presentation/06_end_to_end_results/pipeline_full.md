# End-to-End Autonomous Pick-and-Place Pipeline

## 1. Overview

전체 미션: 드론이 **자율적으로** 페디스탈 위의 물체에 접근 → 파지 → 목표 지점으로 운반.
RL waypoint navigation과 analytical PID docking을 task decomposition으로 결합한다.

```
[Approach]         [Dock]           [Climb]        [Delivery]
RL waypoint  →  PID analytical  →  PID climb  →  RL loaded
navigation       precision          ascent         waypoint
(Stage 1)        docking           (low-gain)      navigation
                 + auto-close                      (Stage 4)
```

---

## 2. Phase State Machine

Headless eval (`eval_mission_headless.py`) 기준, 128개 환경 병렬 실행:

```
SETTLE(0) → APPROACH(1) → DOCK(2) → CLIMB(3) → DELIVERY(5) → ARRIVED(6) → DONE(7)
                                         ↑                                       |
                                         └─── false grasp detected ──────────────┘
```

| Phase | 제어기 | 시간 제한 | 전이 조건 |
|-------|--------|----------|----------|
| 0: Settle | — | 1 step | teleport commit 대기 |
| 1: Approach | RL (Stage 1, 22D/4D) | 60s (9000 steps) | WP 도달 + 안정 |
| 2: Dock | PID analytical (action=0) | 12s (1800 steps) | contain ≥ 325 + grip 확인 |
| 3: Climb | PID climb (Kp=6, Kd=5) | 10s (1500 steps) | drone Z > 1.0m |
| 5: Delivery | RL (Stage 4, 23D/4D) | — | 최종 WP 도달 |
| 6: Arrived | — | 1 step | 성공 기록 |
| 7: Done | — | — | 미션 재시작 |

### Phase 4 (HOVER_STAB)

정의는 되어 있으나 현재 미사용. 향후 RL 모델 warm-up에 활용 가능.

---

## 3. 사용 모델 및 체크포인트

| 역할 | 모델 | Obs | Act | 체크포인트 |
|------|------|-----|-----|-----------|
| Approach | PPO (Stage 1) | 22D | 4D | `logs/gripper_wp_flight_v6/final_agent.pt` |
| Dock | PID analytical | — | — | 코드 내장 (`_compute_analytical_base()`) |
| Climb | PID low-gain | — | — | 코드 내장 |
| Delivery | PPO (Stage 4) | 23D | 4D | `logs/gripper_wp_loaded_v20/best_agent.pt` |

---

## 4. Phase 1: Approach (RL Waypoint Navigation)

### 4.1 미션 설정

- 드론 spawn: box로부터 2-3m 거리, $z = 3.0\,\text{m}$, 랜덤 각도
- Waypoint 생성: `generate_waypoints()` → 3개 중간 WP + 1개 최종 WP (box 상공 0.5m)
- WP 간 거리: 0.8-2.3m, 랜덤 방향 (목표 방향 ±0.75 rad 편차)

### 4.2 관측 구성 (22D)

Stage 1 RL 모델의 입력:

$$
\mathbf{o} = [\mathbf{v}_b,\; \boldsymbol{\omega}_b,\; \text{vec}(\mathbf{R}),\; \mathbf{g}_b,\; \hat{\mathbf{a}}_{t-1}]
$$

- Velocity noise: $\sigma = 0.03$
- Position noise: $\sigma = 0.01$

### 4.3 전이 조건

모든 waypoint 도달 ($d < 0.30\,\text{m}$) 후, 안정 조건 확인:
- 속도 $\|\mathbf{v}\| < 2.0\,\text{m/s}$
- 기울기 $\theta < 30°$
- XY 오차 $d_{xy} < 0.8\,\text{m}$
- 5초 동안 유지 (post_wp_wait)

또는 timeout (60s)이되 최소 접근 조건 충족:
- $d_{xy} < 2.0\,\text{m}$, $z > z_{\text{ground}}$

### 4.4 실패 조건

| 실패 | 조건 |
|------|------|
| approach_timeout_too_far | timeout + XY > 2.0m |
| too_low | $z_{\text{local}} < 0.10\,\text{m}$ |
| too_far | $d_{xy} > 15\,\text{m}$ |
| too_tilted | $\theta > 70°$ |

---

## 5. Phase 2: Dock (PID Analytical Controller)

### 5.1 제어 모드

- `bypass_analytical = False`: PID가 전체 제어 담당
- RL action은 0으로 설정 → residual scale 0.0 → pure analytical
- 그리퍼 open 상태에서 시작

### 5.2 Docking 프로세스

1. **XY 정렬**: 비대칭 PD로 box 중심 추적
2. **Gated descent**: XY 10cm 이내 시 하강 시작
3. **Standoff approach**: 30cm → 13cm → 2cm 단계적 접근
4. **Containment detection**: overlap ratio > 50% 지속 시 contain_hold_count 증가
5. **Gripper close**: contain ≥ 150에서 시작, 175 step 동안 닫기
6. **Grip confirm**: contain ≥ 325 **이고** box-drone 거리 < 0.30m

### 5.3 Grip Miss 처리

Contain ≥ 325이지만 box 거리 ≥ 0.30m → **grip miss**:
- contain_hold_count = 0으로 reset
- 그리퍼 다시 open
- PID가 재시도 (dock timeout 이내)

### 5.4 Mass Update

Grip 확인 시 attitude controller의 질량 갱신:

$$
m_{\text{ctrl}} = m_{\text{base}} \cdot s_{\text{mass}} + m_{\text{payload}}
$$

여기서 $m_{\text{base}} = 1.080\,\text{kg}$, $s_{\text{mass}} \sim U(0.9, 1.1)$, $m_{\text{payload}} = 0.2\,\text{kg}$.

### 5.5 실패 조건

| 실패 | 조건 | 진단 |
|------|------|------|
| dock_timeout | 1800 steps | contain 수준으로 세부 분류 |
| dock_timeout_gripper_closed | contain ≥ 150+ but Z not climbing | grip miss 반복 |
| dock_timeout_partial | contain 50-150 | 불안정 접촉 |
| dock_timeout_xy | XY > 0.15m | XY 정렬 실패 |
| dock_timeout_high | Z > 0.30m above box | 하강 gate 미작동 |
| dock_timeout_stuck | vel < 0.05 m/s | stuck recovery 실패 |
| box_fell_during_dock | box Z < 0.30m, contain < 150 | 접촉 충격으로 box 낙하 |

---

## 6. Phase 3: Climb (PID Low-Gain)

### 6.1 제어

- **PID climb mode** 자동 활성화 (contain ≥ 325)
- 목표: $z_{\text{climb}} = 1.5\,\text{m}$
- 게인: $K_p = 6.0$, $K_d = 5.0$ (정상 모드의 절반)
- **저게인 이유**: $K_p = 12$에서 ±8 m/s² clamp에 도달 → bang-bang 진동 → box 이탈

### 6.2 False Grasp Detection

Headless eval에서 30 step 후 검증:
- Drone Z가 box Z보다 0.25m 이상 위 **이고** box가 페디스탈 높이 근처 → false grasp
- Contain count reset → Phase 2 (Dock)로 회귀

### 6.3 전이 조건

$$
z_{\text{drone}} > 1.0\,\text{m} \implies \text{CLIMB} \to \text{DELIVERY}
$$

### 6.4 실패 조건

| 실패 | 조건 |
|------|------|
| climb_failed (timeout) | 1500 steps, Z < 1.0m |
| climb_failed (crash) | Z < 0.05m |
| climb_failed (box_lost) | box-drone distance > 0.50m |

---

## 7. Phase 5: Delivery (RL Loaded Navigation)

### 7.1 Waypoint 생성

Climb 완료 후 delivery target 및 경유 WP 생성:
- Delivery target: box 위치로부터 XY 2-3m 거리, $z = 2.5\,\text{m}$
- 3개 중간 WP + 1개 최종 WP

### 7.2 관측 (23D)

Stage 4 모델의 입력 — Stage 1과 동일 + payload mass:

$$
\mathbf{o}_{\text{loaded}} = [\mathbf{o}_{22},\; \hat{m}_{\text{payload}}]
$$

$\hat{m}_{\text{payload}} = 0.2 + \epsilon$, $\epsilon \sim \mathcal{N}(0, 0.02^2)$.

### 7.3 Box Drop Detection

매 step 검증:

$$
\text{dropped} = (z_{\text{box,local}} < 0.30\,\text{m}) \vee (d_{\text{box-drone}} > 0.50\,\text{m})
$$

Box drop 시 즉시 미션 실패.

### 7.4 전이 조건

최종 WP 도달 ($d < 0.30\,\text{m}$) → Phase 6 (ARRIVED) → 성공.

---

## 8. Timing Budget

| Phase | 최대 시간 | Steps (150Hz) |
|-------|----------|---------------|
| Settle | 0.007s | 1 |
| Approach | 60s | 9000 |
| Dock | 12s | 1800 |
| Climb | 10s | 1500 |
| Delivery | ∞ (mission limit) | — |
| **총 미션** | **120s** | **18000** |

---

## 9. 성공/실패 정의

### Full Mission Success

1. Approach 완료 (box 상공 도달)
2. Dock 성공 (그리퍼 닫기 + grip 확인)
3. Climb 완료 (Z > 1.0m)
4. Delivery 완료 (최종 WP 도달)
5. **Box 유지** (전 과정에서 drop 없음)

### 부분 성공

- **dock_success**: Phase 1-2 성공 (물체 파지까지)
- **full_success**: Phase 1-5 전체 완료

---

## 10. Sequential vs Headless Eval 차이

| 항목 | eval_mission.py | eval_mission_headless.py |
|------|----------------|------------------------|
| 병렬 환경 | 1 | 128 |
| Dock contain 임계값 | 200 | 325 |
| Stage 4 model | v16 | v20 |
| False grasp detection | 없음 | 30 step 후 검증 |
| Phase 수 | 5 | 8 (settle, hover_stab 포함) |
| 미션 수 | 300 (고정) | 500+ (환경 재활용) |
| 출력 | 미션별 상세 로그 | 3000 step 단위 요약 |

---

## 11. 데이터 파일

- `../data_csv/waypoint_eval_flight_*.json`: 비행 eval JSON
- End-to-end 결과는 스크립트 실행 시 stdout으로 출력
