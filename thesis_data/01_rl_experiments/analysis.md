# RL Experiments Analysis (Stage 3 Grasping)

## Overview

Stage 3의 목표: 드론이 페디스탈 위의 8cm 큐브 박스를 그리퍼로 파지.
그리퍼 구조: 양쪽 plate가 50° 개방, strut 간격 10cm (X), plate tip 간격 20.8cm (Y).
박스 8cm → X 마진 1cm (critical), Y 마진 6.4cm (여유).

## 실험 시리즈 요약

### Series 1: Kinematic Box (4/10)

| Model | Env | Episode | Dock% | 핵심 발견 |
|-------|-----|---------|-------|----------|
| v2 | kinematic, 40s | 40s | 31.2% | 첫 Stage 3 시도 |
| v3 | kinematic, 40s | 40s | 38.5% | v2 연장 학습 |
| v3 | **dynamic**, 40s | 40s | 11.4% | dynamic 전환 시 성능 급락 |

**핵심 발견**: Kinematic→Dynamic 전환 시 dock 38.5%→11.4%. 드론이 박스를 쳐서 밀림.

### Series 2: Phase 1 — Box 위 Spawn (4/10-4/11)

Phase 1: 접근 단계 생략, "박스 위에서 정밀 안착"만 집중.
드론 spawn: 박스 바로 위 (z=0.85m, ±5cm XY), episode 3초.

| Model | 변경사항 | TB overlap | TB r_success | Eval dock% | Eval grasp% |
|-------|---------|-----------|-------------|-----------|------------|
| v1 | Phase 1 환경만 변경 | 0.328 | 0.099 | 24.6% | 17.4% |
| v2 | v1 이어 학습 | 0.342 | 0.108 | 24.8% | 19.4% |
| v3 | target=박스 위 20cm, column_exit | 0.348 | 0.108 | - | - |
| v4 | in_column hard, 단순화 | **0.422** | **0.148** | 23.3% | 18.6% |
| v5 | +r_desc_speed, +r_hover_dock | 0.390 | 0.122 | 21.4% | 11.2% |

**핵심 발견 1: TB vs Eval 괴리**
v4의 TB overlap이 0.422 (+23%)로 역대 최고였지만, eval dock은 23.3%로 v2(24.8%)보다 낮음.
TB 평균값이 "기둥 안 hovering"으로 부풀려진 것. r_success가 더 신뢰성 높은 지표.

**핵심 발견 2: 보상 추가 = 성능 하락**
v4→v5에서 r_desc_speed + r_hover_dock 2개 추가 → eval grasp 19.4%→11.2% 하락.
새 보상이 hovering exploit를 만들어 학습 방해.
**단순화가 추가보다 효과적.**

**핵심 발견 3: PPO 피크 후 하락**
거의 모든 학습에서 50-75% 지점에서 best → 100%에서 하락.
best_agent.pt 자동 저장이 핵심.

### Series 3: Phase 2 전환 시도 (4/11, 모두 실패)

Phase 1 모델을 Phase 2 (z=1.5m spawn)로 전환 시도.

| 시도 | Checkpoint | 변경사항 | 결과 |
|------|-----------|---------|------|
| from v5 | Phase 1 v5 | Phase 2 환경 | 완전 붕괴: xy 25m |
| box 6cm | v3 best | 6cm box + dynamic | 붕괴: 너무 많은 변수 동시 변경 |
| from v4 | Phase 1 v4 | Phase 2 환경 | 붕괴: 비행 능력 상실 |
| baseline | v3 best | 8s + dynamic + 구 보상 | 붕괴: v3는 40s 학습 |

**핵심 발견**: Phase 1 모델은 box 위 30cm spawn에서만 학습돼서 비행 능력 자체를 상실.
1.5m spawn에서 박스를 못 찾음. **Curriculum 전환의 구조적 한계.**

### Series 4: Residual RL (4/14, 실패)

PD analytical controller(49%) 위에 RL residual(scale=0.1) 추가 시도.
보상: outcome 기반 (r_descent_precision, r_soft_contact, r_x_precision).

| 지표 | 시작 (PD only) | 5% 학습 후 |
|------|---------------|-----------|
| overlap | 0.042 | 0.004 (↓10x) |
| r_success | 0.009 | 0.001 (↓9x) |
| r_contain | 0.999 | 0.094 (↓10x) |
| Total reward | 3811 | 5163 (↑) |

**Total reward가 올라갔지만 dock 지표가 전부 하락.**
RL이 dock 대신 hovering으로 r_approach + r_x_precision + r_soft_contact를 수집.
**Hovering exploit**: dock을 시도하지 않는 게 step당 보상이 더 높았음.

## RL 실패의 근본 원인 분석

### 1. Dynamic Box의 Contact Dynamics
- Kinematic: dock 38.5% → Dynamic: dock 11.4%
- 그리퍼가 박스를 쳐서 밀림 → RL이 "부드럽게 안착"을 학습해야 하지만 보상 신호가 너무 sparse
- Contact 순간의 보상 차이가 작아서 credit assignment 실패

### 2. Reward Shaping의 딜레마
- Sparse reward (r_success만): 학습 너무 느림 (225 step 유지 필요)
- Dense reward (r_approach 등): hovering exploit 유발
- **어떤 intermediate reward를 넣어도 "dock 안 하고 hovering"이 더 높은 보상**

### 3. 탐색 비효율
- PPO random exploration: action이 상쇄 → 30 step 누적 효과 없음
- Residual scale 0.1: step당 0.02mm position change → 보상 변화 0.012/step
- 학습 신호가 noise 수준

### 4. Curriculum 전환 실패
- Phase 1 (3s, box-above spawn) → Phase 2 (8s, 1.5m spawn): 모든 시도 실패
- Phase 1에서 학습한 "정밀 안착" 스킬이 Phase 2의 "비행+접근" 요구와 양립 불가
- 비행 능력 상실: Phase 1 모델은 30cm 범위에서만 동작

## 데이터 파일
- `training_log_original.md`: 전체 학습 로그 (시간순)
- `../data_csv/rl_training_summary.csv`: TB 메트릭 요약
- `../data_csv/rl_eval_results.csv`: Eval 결과 모음
