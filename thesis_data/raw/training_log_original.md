# Gripper-Drone PPO Training Log

> 모든 시도와 결과를 시간순으로 정리. Stage 1 재학습 → Stage 3 → Phase 1 v1~v5.

---

## Stage 1: Basic Flight (재학습)

### 목표
드론이 랜덤 3D 목표 지점에 도달하고 호버링 유지.

### 환경 설정
| 파라미터 | 값 |
|---------|-----|
| stage | BASIC_FLIGHT |
| episode_length_s | 6.0 |
| spawn_z | 3.0m (±0.5) |
| goal_pos_range_xy | ±1.0m |
| goal_pos_range_z | (2.0, 4.0)m |
| lock_gripper | True |
| num_envs | 4096 |

### 보상 구조 (`reward_fn.py: compute_stage1_rewards`)

```python
r_pos     = 4.0 * exp(-1.2 * pos_err)               # 거리 기반 보상
r_arrive  = 10.0 * (pos_err < 0.2)                  # 도달 보너스 (binary)
r_hover   = 5.0 * (pos_err < 0.2) * exp(-3.0 * vel_norm)  # 도달 후 정지 보상
r_smooth  = -0.1 * ‖Δaction‖²                       # 행동 부드러움
r_mag     = -0.02 * ‖action‖²                       # 행동 크기

total = r_pos + r_arrive + r_hover + r_smooth + r_mag
```

**핵심 설계:** "도달 보너스 + 도달 유지 보상"으로 정밀 도달 강제.

### 학습 결과

| 모델 | overlap | r_arrive | r_hover | reward |
|-----|---------|----------|---------|--------|
| logs/stage1_ppo_v2 (best) | - | 5.37 | 1.25 | 7609 (best) |

**Eval 결과:**
```
Episodes:          100
Mean reward:       9851 ± 4172
Crash rate:        15.0%
Min pos error:     0.15m (avg of per-episode best)
Reached <0.5m:     94%
Reached <0.3m:     86%
Reached <0.1m:     62%  ← 정밀 도달 비율
```

### 학습 패턴
- 75%에서 피크 → 100%에서 하락 (전형적 PPO 피크 후 하락)
- best checkpoint는 75% 시점에 저장됨

---

## Stage 3: Grasping

### 목표
드론이 박스를 그리퍼 사이에 정확히 안착시키고 자동 닫기로 파지.

### 환경 설정 (공통)

| 파라미터 | 값 |
|---------|-----|
| stage | GRASPING |
| episode_length_s | 8.0 (Phase 2), 3.0 (Phase 1) |
| box | 8cm 큐브, 0.2kg, kinematic (v3 이전) / dynamic (Phase 1+) |
| pedestal | 30×30×50cm, kinematic |
| dock_threshold | 150 (1초 누적) → 자동 그리퍼 닫기 |
| contain_success | 225 (1.5초 누적) → 성공 + truncation |

### Stage 3 학습 시리즈

#### v2 (`stage3_ppo_v2`)

**보상 구조:**
- 축 1: r_approach_xy(3.0) + r_approach_z(2.0 * above_box)
- 축 2: r_x_align/y_align/z_below (soft gate)
- r_z (staged descent), r_fine, r_contain, r_dock_bonus, r_hold_stable, r_success
- r_tilt_descent, r_tilt_dock, r_smooth, r_mag

**Eval 결과 (kinematic):**
```
Dock success: 31.2% (756/2420)
XY err at dock: 3.4cm
```

#### v3 (`stage3_ppo_v3`)

**v2 대비 변경:**
- 같은 보상 구조에서 학습 연장

**Eval 결과 (kinematic):**
```
Dock success: 38.5% (968/2515)
XY err at dock: 3.4cm
```

**Eval 결과 (dynamic 박스 테스트):**
```
Dock success: 11.4% (355/3120)
Grasp success: 10.0% (312/3120)
Far: 47%, Near TO: 30%
```
→ Dynamic 환경에서는 성능 급락. 드론이 박스를 쳐서 밀림.

---

## Stage 3 Phase 1 시리즈 (Dynamic 박스 + 드론 박스 위 스폰)

### Phase 1 환경 설정

| 파라미터 | 값 |
|---------|-----|
| grasping_phase | 1 |
| episode_length_s | 3.0 |
| drone spawn | 박스 바로 위 (z=0.85m, ±5cm XY) |
| box | dynamic (kinematic_enabled=False) |

**의도:** 접근 단계 생략, "박스 위에서 정밀 안착"만 집중 학습.

### Phase 1 v1 (`stage3_phase1`)

**보상 구조:** v3 그대로 (변경 없음, 환경만 Phase 1)

**TB 결과:**
- overlap best: 0.328
- r_success best: 0.099

**Eval (Phase 1, dynamic):**
```
Dock success: 24.6% (1292/5254)
Grasp success: 17.4% (916/5254)
Avg dock time: 1.4s
XY err: 3.4cm
Far: 43%, Near TO: 50%
```

### Phase 1 v2 (`stage3_phase1_v2`)

**v1 대비 변경:** v1 best에서 이어 학습 (환경/보상 동일)

**TB 결과:**
- overlap best: 0.342
- r_success best: 0.108

**Eval:**
```
Dock success: 24.8% (1262/5087)
Grasp success: 19.4% (985/5087)
XY err: 3.2cm
Far: 52%, Near TO: 43%
```

### Phase 1 v3 (`stage3_phase1_v3`) — 보상 재설계 1차 시도

**v2 대비 변경:**
- 축 1: `r_approach = 4.0 * exp(-1.5 * target_err)` (target = 박스 위 20cm)
- 축 2: `r_in_column(8.0)`, `r_column_center(4.0)`, `r_column_exit(-5.0)`
- 축 3: r_fine 제거
- r_z (staged descent) 제거

**TB 결과:**
- overlap best: 0.348 (한계점 근처)
- r_success best: 0.108

**문제:**
- 축 1 (박스 위 20cm 목표) → 드론이 20cm 위 hovering 유도
- 축 2 r_column_exit → 박스 근처 회피 학습
- 피크 후 하락 발생

### Phase 1 v4 (`stage3_phase1_v4`) — 보상 재설계 2차

**v3 대비 변경:**
- 축 1 복원: `r_approach_xy(3.0)` (박스 직접 접근)
- r_column_exit 제거
- r_dock_bonus 제거 (r_contain과 중복)
- r_smooth: 0.2 → 0.1

**TB 결과 (역대 최고):**
- overlap best: 0.422 (이전 0.342 → +23%)
- full_contain best: 0.457
- r_success best: 0.148 (+37%)
- in_column best: 0.621 (60%+ 기둥 안)
- **단조 증가, 피크 후 하락 없음**

**Eval:**
```
Dock success: 23.3% (1348/5782)
Grasp success: 18.6% (1078/5782)
XY err: 3.2cm
Far: 47%, Near TO: 47%
Crash: 3% (대폭 감소)
```

**핵심 발견:** TB 지표는 +23~37% 증가, 하지만 eval dock은 비슷.
→ TB 지표(평균값)가 "기둥 안 호버링"으로 부풀려진 것.
→ 진짜 안착(overlap > 50% 1초 유지)은 비슷한 수준.

### Phase 1 v5 (`stage3_phase1_v5`) — 하강 속도 + 호버 보상

**v4 대비 변경:**
- `r_desc_speed`: 박스 위 30cm 이내에서 desired_vz 추종
  ```python
  desired_vz = -1.0 * alt_above_box.clamp(max=0.3)
  r_desc_speed = 3.0 * exp(-3.0 * vz_err) * near_box_z
  ```
- `r_hover_dock`: 안착 후 정지 보상
  ```python
  r_hover_dock = 10.0 * is_contained * exp(-5.0 * vel_mag)
  ```

**TB 결과 (악화):**
- overlap best: 0.390 (v4 0.422보다 낮음)
- r_success best: 0.122 (v4 0.148보다 낮음)
- 피크 후 하락 (50% → 100% 급락)

**Eval:**
```
Dock success: 21.4% (1132/5299)
Grasp success: 11.2% (593/5299)  ← 악화
Crash: 6%, Far: 55%, Near TO: 36%
```

**문제 분석:**
- r_desc_speed가 hovering exploit 가능 (alt=0에서 vel=0이 max)
- r_hover_dock는 is_contained 조건이라 학습 신호 희소
- 새 보상이 학습 가속 대신 방해

---

## 핵심 발견 및 교훈

### 1. TB vs Eval 괴리

| 모델 | TB overlap | TB r_success | Eval dock | Eval grasp |
|-----|-----------|--------------|-----------|-----------|
| Phase 1 v2 | 0.342 | 0.108 | 24.8% | 19.4% |
| Phase 1 v4 | 0.422 (+23%) | 0.148 (+37%) | 23.3% | 18.6% |
| Phase 1 v5 | 0.390 | 0.122 | 21.4% | 11.2% |

**TB 평균 지표가 "기둥 안 체류 시간"으로 부풀려질 수 있음.** r_success가 가장 신뢰성 높은 지표.

### 2. PPO 피크 후 하락 패턴

거의 모든 학습에서 발생:
- best checkpoint가 50~75% 지점에서 저장됨
- 학습 끝까지 가면 best보다 나빠짐
- best_agent.pt 자동 저장이 핵심

### 3. 보상 변경의 위험

- 보상 구조 변경 후 이어 학습 시 value function 재적응 실패 빈번
- v3 → v4 (단순화 + 중복 제거): 성공적 개선
- v4 → v5 (새 보상 추가): 실패
- 핵심: **단순화가 추가보다 효과적**

### 4. Dynamic 박스의 의미

- Kinematic eval: dock 38.5%
- Dynamic eval: dock 11.4%, grasp 10%
- 격차: "박스를 치는 행동"이 kinematic에서는 페널티 없음
- Dynamic으로 학습해야 진짜 정밀 안착 학습 가능

### 5. 렌더링에서 본 핵심 문제

```
정밀도: Reached <0.1m: 88-90% (정확히 도달)
하지만 Crash rate: 68-90%
```

문제:
- 한쪽 그리퍼가 먼저 박스 모서리/페디스탈 면에 닿음
- 충격으로 드론 기울어짐 → crash
- 안착 시 충분히 감속 못함

**필요한 행동:** 박스 표면에 착륙 X, 그리퍼 사이에 박스 호버링.

---

## 현재 보상 구조 (v5, drone_env.py L441-521)

```python
# 축 1: XY 접근
r_approach = 3.0 * exp(-1.5 * xy_err)

# 축 2: 기둥 영역 (hard boundary)
in_column = (
    (|box_local_x| < 0.05) &
    (|box_local_y| < 0.104) &
    (box_local_z < 0.02) &
    (box_local_z > -0.40)
)
r_in_column = 8.0 * in_column
x_center_factor = (1 - |box_local_x|/0.05).clamp(0,1)
y_center_factor = (1 - |box_local_y|/0.104).clamp(0,1)
r_column_center = 4.0 * in_column * x_center_factor * y_center_factor

# 축 3: 정밀 안착
r_contain = 15.0 * overlap_ratio + 10.0 * full_contain
is_contained = (overlap_ratio > 0.50)
hold_duration = (contain_hold_count / 150).clamp(max=3.0)
r_hold_stable = 10.0 * hold_duration * is_contained
r_success = 50.0 * (contain_hold_count >= 225)

# 속도 제어
desired_vz = -1.0 * alt_above_box.clamp(max=0.3)
r_desc_speed = 3.0 * exp(-3.0 * |vel_z - desired_vz|) * near_box_z
r_hover_dock = 10.0 * is_contained * exp(-5.0 * vel_mag)

# 페널티
r_tilt_descent = -3.0 * tilt * in_column
r_tilt_dock = -2.0 * tilt * (overlap > 0.3)
r_smooth = -0.1 * ‖Δaction‖²
r_mag = -0.1 * ‖action‖²

total = (r_approach + r_column_total + r_contain + r_hold_stable + r_success
       + r_desc_speed + r_hover_dock + r_tilt_descent + r_tilt_dock
       + r_smooth + r_mag)
```

## 기둥 영역 정의

**기둥 = 그리퍼 plate 끝(50° 개방)이 이루는 사각형을 아래로 무한히 투영한 사각 기둥**

```
좌표계: gripper local frame (드론 회전 적용)

X폭: ±0.05m (10cm) — strut 간격
Y폭: ±0.104m (20.8cm) — plate tip 위치
Z범위: -0.40m ~ +0.02m (그리퍼 바로 아래로 40cm)
```

**박스 (8×8×8cm)는 X 방향으로 거의 딱 맞음 (8cm < 10cm), Y 방향은 여유 있음 (8cm << 20.8cm).**

---

---

## Phase 2 전환 시도들 (실패)

Phase 1 모델들이 박스 위 30cm spawn에서만 학습되어 멀리서 접근하는 비행 능력을 잃었음.
Phase 2 (드론이 z=1.5m에서 spawn, 기존 Stage 3 환경)로 전환을 여러 번 시도.

### Phase 2 from v5 best (`stage3_phase2_from_v5`)

**환경:** Phase 2, 8초, dynamic, 8cm 박스, v5 보상

**결과:** 23% 시점에 overlap 0.0024, xy_err 25m → 완전 붕괴
- v5 best (Phase 1 학습)이 1.5m spawn에서 박스를 못 찾음
- 비행 능력 자체를 상실한 상태

### Phase 2 box 6cm (`stage3_phase2_box6cm`)

**환경:** Phase 2, 8초, dynamic, **6cm 박스** (충돌 마진 2배), v5 보상
**Checkpoint:** v3 best

**결과:** 10% 시점에 overlap 0.011, xy_err 65m → 붕괴
- v3 best는 8cm 박스 + 구 보상으로 학습
- 박스 크기 + dynamic + 새 보상 동시 변경 → 재적응 실패

### Phase 2 from v4 best (`stage3_phase2_from_v4`)

**환경:** Phase 2, 8초, dynamic, 8cm, v4 보상 (r_desc_speed/r_hover_dock 제거)
**Checkpoint:** Phase 1 v4 best

**결과:** 3% 시점에 overlap 0.0008, xy_err 17m → 붕괴
- Phase 1 v4도 1.5m spawn에서 박스 못 찾음
- Phase 1 환경(박스 위 30cm)에서만 학습된 한계

### Phase 2 from v3 (구 보상 복원, `stage3_phase2_baseline`)

**환경:** Phase 2, 8초, dynamic, 8cm
**Checkpoint:** v3 best (`stage3_ppo_v3`)
**보상:** v3 시점 보상 복원 (r_approach_xy/z, r_x_align/y_align/z_below soft gate, r_z staged, r_fine, r_dock_bonus 등)

**결과:** 6% 시점에 overlap 0.015, r_contain 0.36 → 망가지는 중
- v3 best는 40초 episode로 학습됨
- 8초 환경에서 시간 부족 + dynamic 박스 적응 실패

---

## Stage 3 8s Kinematic Baseline (현재 진행 중) — `stage3_8s_kinematic`

### 핵심 결정

**v3 best (40초 학습 모델) 사용 중단.** 8초 환경과 호환 안 됨.
**Stage 1 best (`stage1_ppo_v2`)에서 처음부터 8초로 학습.** 변수를 최소화:
- kinematic 박스 (안정적, dynamic은 나중에)
- 8초 episode
- Phase 옵션 없음 (drone z=1.5 spawn, 기본 Stage 3)
- v3 보상 (방금 복원)

### 학습 커맨드

```bash
python train.py --stage 3 --num_envs 4096 --max_steps 1000000000 \
    --checkpoint logs/stage1_ppo_v2/best_agent.pt \
    --log_dir logs/stage3_8s_kinematic
```

### 환경 설정

| 파라미터 | 값 |
|---------|-----|
| stage | GRASPING (`--phase` 없음) |
| episode_length_s | 8.0 |
| 박스 | 8cm, kinematic |
| 페디스탈 | 30×30×50cm |
| drone spawn | z=1.5m + spawn_offset(±0.5m XY/Z) |
| dock_threshold | 150 (1초 누적) |
| contain_success | 225 (1.5초 누적) |
| max_steps | 1B |

### 보상 구조 (v3 복원)

```python
# 축 1
r_approach_xy = 3.0 * exp(-1.5 * xy_err)
r_approach_z  = 2.0 * exp(-2.0 * z_err) * above_box

# 축 2
approach_gate = ((0.25 - xy_err) / 0.20).clamp(0, 1)
r_x_align = 1.5 * x_contain * approach_gate
r_y_align = 1.5 * y_contain * approach_gate
r_z_below = 2.0 * z_below * approach_gate
xy_aligned = (xy_err < 0.10)
alt_target = box_z + 0.25 * (1 - xy_aligned)
r_z = 2.0 * exp(-2.0 * |gripper_z - alt_target|)
r_fine = 5.0 * exp(-10 * xy_err) * approach_gate * above_box

# 축 3
r_contain = 15 * overlap + 10 * full_contain
r_dock_bonus = 10 * is_contained
r_hold_stable = 10 * hold_duration * is_contained
r_success = 50 * (contain_hold_count >= 225)

# 페널티
r_tilt_descent = -3.0 * tilt * xy_aligned
r_tilt_dock = -2.0 * tilt * (overlap > 0.3)
r_smooth = -0.2 * ‖Δaction‖²
r_mag = -0.1 * ‖action‖²
```

### 학습 진행 상황 (6% 시점)

| 지표 | 시작 | 6% |
|-----|------|----|
| overlap | 0.001 | 0.025 |
| r_contain | 0.025 | 0.589 |
| r_dock_bonus | 0.009 | 0.218 |
| r_success | 0.0002 | 0.006 |
| r_approach | 2.69 | 3.23 |
| r_column | 0.19 | 1.36 |
| xy_err | 0.37 | 0.21 |
| reward | 1845 | 19366 |

**정상 학습 중.** Stage 1 best가 8초 + Stage 3 환경에 정상 적응. 첫 프레임 outlier 없음.
이번 학습이 **8초 + dynamic 박스 적응의 baseline**이 될 예정.

### 다음 단계 (계획)

1. **현재 학습 완료 → eval로 8초 kinematic baseline 측정**
2. **Hybrid 접근 검토:** RL (멀리서 접근) + 분석적 제어 (정밀 docking)
   - RL이 정밀 안착을 학습하기 어려움 (모든 시도가 70-90% crash)
   - 거리 비행은 RL, 마지막 5cm 안착은 PID/MPC
3. **또는 Dynamic 박스 학습 (현재 baseline에서 이어 학습)**

---

## 시간순 시도 요약 (4/10 ~ 4/11)

```
4/10:
  ✓ stage1_ppo_v2 (Stage 1 재학습, 도달 보너스 + hover 보상)
    eval: dock 62% (<0.1m), crash 15%

  ✓ stage3_ppo_v2 (Stage 1 best → Stage 3, 40초 episode, kinematic)
    eval: dock 31.2%

  ✓ stage3_ppo_v3 (v2 → v3, 학습 연장, 40초)
    eval kinematic: dock 38.5%
    eval dynamic: dock 11.4%, grasp 10.0%

4/10 후반~4/11:
  episode 8.0초로 변경

  ✓ stage3_phase1 (v3 → Phase 1, 박스 위 spawn, 3초, dynamic)
    eval: dock 24.6%, grasp 17.4%

  ✓ stage3_phase1_v2 (v1 이어 학습)
    eval: dock 24.8%, grasp 19.4%

  ✗ stage3_phase1_v3 (보상 재설계 1차: target=박스 위 20cm)
    실패 — hovering exploit, 피크 후 하락

  ✓ stage3_phase1_v4 (보상 재설계 2차: in_column hard boundary, 단순화)
    TB best: overlap 0.422, r_success 0.148
    eval: dock 23.3%, grasp 18.6%
    "TB vs eval 괴리" 발견

  ✗ stage3_phase1_v5 (r_desc_speed + r_hover_dock 추가)
    실패 — 새 보상이 학습 방해
    eval: dock 21.4%, grasp 11.2%

4/11 (Phase 2 전환 시도):
  ✗ stage3_phase2_from_v5 (Phase 1 v5 → Phase 2)
    실패 — Phase 1 모델은 1.5m spawn에서 박스 못 찾음

  ✗ stage3_phase2_box6cm (v3 best, 박스 6cm + dynamic + 새 보상)
    실패 — 너무 많은 변수 동시 변경

  ✗ stage3_phase2_from_v4 (Phase 1 v4 → Phase 2)
    실패 — Phase 1 모델 비행 능력 상실

  ✗ stage3_phase2_baseline (v3 best, 8초 + dynamic + 구 보상)
    실패 — v3는 40초 학습이라 8초 환경 적응 못 함

4/11 현재:
  ▶ stage3_8s_kinematic (Stage 1 best → 8초 + kinematic + Stage 3)
    Stage 1 best에서 처음부터 8초 학습, 변수 최소화
    6% 시점 정상 학습 중
```
