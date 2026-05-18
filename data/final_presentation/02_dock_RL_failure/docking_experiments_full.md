# Docking Experiments: RL Attempts, Analytical Comparison, & Hybrid Approaches

> 이 문서의 실험 데이터는 2026-04-10~14 시점 기준.
> 당시 analytical controller는 PD였으며, 이후 Z축 integral 추가로 현재 PID.
> 현재 시스템에서 Stage 3 도킹은 PID를 사용한다 (→ `01_system_and_controller.md`).

---

## Part A: RL Docking Experiments (Stage 3)

### A.1 Overview

Stage 3 목표: 드론이 페디스탈 위 8cm 큐브를 그리퍼로 파지.
그리퍼 구조: plate 50° 개방, strut 간격 10cm (X), plate tip 간격 20.8cm (Y).
X 마진 1cm (critical), Y 마진 6.4cm (여유).

### A.2 Series 1: Kinematic Box (4/10)

| Model | Env | Dock% | 핵심 발견 |
|-------|-----|-------|----------|
| v2 | kinematic, 40s | 31.2% | 첫 Stage 3 시도 |
| v3 | kinematic, 40s | 38.5% | v2 연장 학습 |
| v3 | **dynamic**, 40s | 11.4% | dynamic 전환 시 성능 급락 |

**핵심**: Kinematic→Dynamic 전환 시 dock 38.5%→11.4%. 드론이 박스를 쳐서 밀림.

### A.3 Series 2: Phase 1 — Box 위 Spawn (4/10-4/11)

접근 생략, "박스 위에서 정밀 안착"만 집중. Spawn: 박스 위 (z=0.85m, ±5cm XY), 3초.

| Model | 변경사항 | TB overlap | Eval dock% | Eval grasp% |
|-------|---------|-----------|-----------|------------|
| v1 | Phase 1 환경 | 0.328 | 24.6% | 17.4% |
| v2 | v1 연장 | 0.342 | **24.8%** | **19.4%** |
| v4 | in_column hard, 단순화 | **0.422** | 23.3% | 18.6% |
| v5 | +r_desc_speed, +r_hover_dock | 0.390 | 21.4% | 11.2% |

**핵심 발견**:
1. **TB vs Eval 괴리**: v4 TB overlap 최고(0.422)이나 eval은 v2보다 낮음. "기둥 안 hovering"으로 TB 부풀려짐
2. **보상 추가 = 성능 하락**: v4→v5 보상 2개 추가 → grasp 19.4%→11.2%. Hovering exploit 유발
3. **PPO 피크 후 하락**: 50-75% 지점에서 best → 100%에서 하락. best_agent.pt 자동 저장 필수

### A.4 Series 3: Curriculum Transfer (4/11, 전수 실패)

Phase 1 → Phase 2 (z=1.5m spawn) 전환 시도:

| 시도 | 결과 |
|------|------|
| Phase 1 v5 → Phase 2 | 완전 붕괴: xy 25m |
| Phase 1 v4 → Phase 2 | 비행 능력 상실 |
| v3 best → 8s + dynamic | 붕괴: 환경 불일치 |

**핵심**: Phase 1 모델은 30cm 범위에서만 학습 → 1.5m spawn에서 비행 능력 자체 상실.

### A.5 Series 4: Residual RL (4/14, 실패)

Analytical controller(49%) 위에 RL residual(scale=0.1) 추가.

| 지표 | 시작 (Analytical only) | 5% 학습 후 |
|------|----------------------|-----------|
| overlap | 0.042 | 0.004 (↓10x) |
| r_success | 0.009 | 0.001 (↓9x) |
| Total reward | 3811 | 5163 (↑) |

**Total reward 상승 + dock 지표 하락** = hovering exploit.
Dock 시도의 risk (box_fell terminate) > hovering의 안전한 11.5/step.

---

## Part B: RL Failure Root Cause Analysis

### B.1 Dynamic Box Contact Dynamics
- Kinematic: 38.5% → Dynamic: 11.4%
- 접촉 순간 미세한 XY 오차(3.7cm vs 6.8cm)가 성공/실패 결정
- 보상 signal 차이가 너무 약해 credit assignment 실패

### B.2 Reward Shaping Dilemma
- **Sparse** (r_success만): 225 step 유지가 너무 rare → 학습 불가
- **Dense** (r_approach 등): hovering exploit → dock 안 하는 게 최적
- **어떤 intermediate reward도** "dock 안 하고 hovering"이 더 높은 보상

### B.3 Exploration 비효율
- PPO random: step별 random action 상쇄 → 누적 효과 없음
- Residual scale 0.1: step당 0.02mm position change → noise 수준

### B.4 Curriculum Transfer 구조적 한계
- Phase 1에서 "정밀 안착"과 Phase 2의 "비행+접근"이 양립 불가
- Analytical은 모든 거리에서 동일 법칙 (gain scheduling으로 자연 전환)

---

## Part C: RL vs Analytical Comparison

### C.1 성능 비교표

| 접근법 | Dock% | Crash% | 학습 시간 | 환경 |
|--------|-------|--------|----------|------|
| RL v3 (kinematic) | 38.5% | - | ~6h | kinematic |
| RL Phase1 v2 (best) | 24.8% | ~3% | ~3h | dynamic, box-above |
| **Analytical initial** | **24.7%** | **0%** | **0** | dynamic |
| **Analytical optimized** | **63.8%** | **0%** | **0** | dynamic |
| RL residual on Analytical | <6% | 0% | ~1h | dynamic |

### C.2 접촉 속도 비교 (핵심 데이터)

| | RL | Analytical |
|-|-----|-----------|
| $v_z$ at contact | $-1.1\,\text{m/s}$ | $-0.15\,\text{m/s}$ |
| Dynamic 성능 하락 | 64% | 16% |

### C.3 Task Decomposition: 각 접근법의 강점

| 능력 | RL | Analytical |
|------|----|-----------| 
| 장거리 waypoint navigation | **우수** | 미구현 |
| 적재 비행 적응 | **우수** (23D obs) | 고정 게인 |
| Precision contact | 미흡 (24.8%) | **우수** (63.8%) |
| 안전성 (0% crash) | 미흡 (3-6%) | **보장** |

→ End-to-end: **RL approach → Analytical dock → RL delivery**

---

## Part D: Hybrid Attempts (Historical, 4/14)

### D.1 RL Residual (scale=0.1) — Hovering Exploit

Outcome-based 보상(r_descent_precision, r_soft_contact, r_x_precision) 설계.
is_descending gate로 hover exploit 방지 시도했으나,
r_approach(1.0) + r_x_precision(4.0) + r_column(6.5) = 11.5/step을
hovering으로 수집 가능 → dock 시도보다 안전한 11.5/step 선택.

### D.2 INDI — 구조적 발산

| 설정 | Dock% | 문제 |
|------|-------|------|
| α=0.3, gain=1.0 | 6.0% | v_z > 0 (상승!), tilt 18-25° |
| α=0.1, gain=0.3 | 0.1% | 여전히 발산 |

**원인**: Outer loop INDI 보정 주기 (6.7ms) < inner loop attitude 응답 (13ms) → 양성 피드백.

### D.3 과거 PID — Integral Overshoot

| 설정 | Dock% | 변화 |
|------|-------|------|
| Ki=2.0, clamp ±0.3 | 55.8% | -2.1% |
| Ki=1.0, clamp ±0.2 | 53.2% | -4.7% |
| Ki=1.5, conditional | 54.7% | -3.2% |

**원인**: Approach 중 integral 한 방향 누적 → 도착 시 overshoot.
현재 해결: gated descent (approach 중 Z error ≈ 0) + clamp ±1.0 + leak 0.95.

---

## Data Files

- `data_csv/rl_training_summary.csv`: RL TB 메트릭 (8 experiments)
- `data_csv/rl_eval_results.csv`: RL eval 결과 (14 entries)
- `data_csv/pd_tuning_progression.csv`: Analytical 튜닝 과정
- `data_csv/dock_vs_overlap_lost_at_first_overlap.csv`: 성공/실패 접촉 비교
- `data_csv/learning_curve_*.csv`: 각 실험별 학습 곡선
- `data_csv/reward_design_comparison.csv`: 보상 설계 비교
- `01_rl_experiments/training_log_original.md`: 전체 학습 로그 (시간순 raw)
