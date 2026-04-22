# Thesis Contribution Summary

## Title
**Autonomous Aerial Pick-and-Place with a Rigid Dual-Purpose Landing Gear: A Hybrid RL-Analytical Approach**

## Core Contribution

### Problem
기존 배달 드론은 **사람이 물건을 적재**하고, 드론은 A→B 단방향 운송만 수행.
반품, 부품 조달 등 **양방향 왕복 운송**에서는 드론이 스스로 물건을 집고 내려놓는 능력이 필요하지만,
소형 드론에 별도 그리퍼를 장착하면 페이로드/비행시간이 크게 감소.

### Solution
**Landing gear를 gripper로 겸용하는 Dual-Purpose 설계** + **자율 end-to-end 운송 파이프라인**:
1. 추가 하드웨어 없이 landing gear 자체가 물체를 파지
2. Approach → Dock → Grasp → Transport → Delivery 전체 사이클 자율 수행
3. 공장 간 부품 왕복 운송 등 반복적 운송 작업에 적용 가능

### Technical Contributions

#### 1. Dual-Purpose Landing Gear Design
- Landing gear의 strut(지지대)와 plate(발판)가 gripper의 finger 역할
- 별도 그리퍼 대비 중량 증가 없음 → 비행 성능 유지
- Isaac Lab 시뮬레이션에서 물리 기반 검증

#### 2. End-to-End Autonomous Transport Pipeline
```
[Approach]        [Dock]           [Grasp]      [Transport]      [Delivery]
RL waypoint  →  PD analytical  →  Auto-close  →  RL loaded  →  RL waypoint
navigation       precision          on hold       flight         + release
                 docking            detection
```
- 각 phase에 최적 제어 방식 적용 (task decomposition)
- Phase 전환은 state-based 자동 switching

#### 3. Systematic RL vs Analytical Control Comparison for Precision Docking
- **RL (PPO)**: 8가지 reward 변형 체계적 실험
  - Baseline → sigmoid threshold → safety bonus → mult 강화 → kinetic penalty → xy gate
  - Best: 50.4% dock (kinematic), 18.6% (dynamic)
- **Analytical PD**: adaptive gain scheduling + stuck detection
  - 63.8% dock (dynamic), 0% crash
- **핵심 발견**: RL의 precision docking 한계 메커니즘 규명
  - 접촉 순간 v_z: RL -1.1 m/s vs PD -0.15 m/s (7x 차이)
  - RL 성공/실패 episode의 접촉 조건이 거의 동일 (운에 의존)
  - Dynamic box에서 RL 성능 64% 급락 vs PD 16% 하락
  - Reward shaping의 구조적 한계: 접근 속도 제어와 정밀 안착의 trade-off

#### 4. Hybrid Architecture: Strengths of Both Approaches
- **RL이 잘하는 것**: 장거리 waypoint navigation, 장애물 회피, 적재 상태 비행 적응
- **Analytical이 잘하는 것**: precision contact, 안전한 접근 속도, 0% crash
- Task decomposition으로 각각의 강점 활용

## Application Scenario
**공장 간 부품 자율 왕복 운송:**
- 드론이 A 공장의 부품 거치대에서 자율 pickup
- B 공장으로 운송 후 delivery
- 빈 상태로 A로 복귀 → 반복
- 사람 개입 없이 지속적 운송 작업 수행

## Experimental Results Summary

### Stage 3 Docking (Core Task)

| Approach | Box Type | Dock Rate | Crash | Key Metric |
|----------|----------|-----------|-------|------------|
| RL baseline (PPO) | Kinematic | 31.9% | ~70% | v_z = -1.4 m/s at contact |
| RL + safety bonus | Kinematic | **50.4%** | 12% | v_z = -1.1 m/s |
| RL + safety bonus | Dynamic | 18.6% | 10% | 64% performance drop |
| PD analytical | Dynamic | **63.8%** | **0%** | v_z = -0.15 m/s |

### RL Reward Design Iterations (8 experiments)

| Experiment | Change | Result | Lesson |
|------------|--------|--------|--------|
| 8s_kinematic | Baseline (Stage1→Stage3) | 31.9% dock | Starting point |
| 8s_sigmoid | Sigmoid staged descent | +14% overlap (TB) | Smooth transition helps |
| safety_v1 (mult=2.0) | Safety bonus on dock rewards | **50.4% dock** | Best RL result |
| safety_v2 (mult=3.0) | Stronger multiplier | 41.7% dock (-17%) | Overlap 전 gradient 부재 |
| safety_v3 (xy_quality) | Gate r_approach_z on XY | Collapse | r_z bypass → dive |
| kinetic_v1 (-12) | Kinetic energy penalty | No effect | Gradient too weak |
| kinetic_v2 (-20) | Stronger penalty | r_success -37% | Slow descent → XY drift |
| dynamic_v2 | From scratch, dynamic | 12.7% dock | Clean approach but slow learning |

### End-to-End Mission
- Pipeline: RL approach → PD dock → RL delivery
- Full mission success rate: *improving (target: 30%+ by May)*

## Experimental Environment
- **Simulator**: Isaac Lab (Isaac Sim 4.5), GPU-accelerated
- **Robot**: Custom gripper-drone (URDF, 1.08kg)
- **Object**: 8cm cube, 0.2kg, dynamic rigid body
- **RL**: PPO (SKRL library), 4096 parallel environments
- **Control rate**: 150Hz (policy), 300Hz (physics)
- **Domain randomization**: mass +-10%, motor kf +-15%, wind 0.5N, position noise 2cm

## Data & Code
- `thesis_data/`: 논문 작성용 정리된 데이터
  - `01_rl_experiments/`: RL 실험 분석 + eval raw JSON
  - `02_pd_controller/`: PD 제어기 분석 + eval raw JSON
  - `03_rl_vs_pd_comparison/`: 비교 분석
  - `04_hybrid_attempts/`: hybrid 시도 분석
  - `data_csv/`: 학습 곡선, 비교 테이블 CSV
- `envs/drone_env.py`: 환경 + analytical controller + reward
- `scripts/eval_*.py`: 평가 스크립트들
- `logs/`: 학습된 모델 checkpoints (140+)
- `runs/`: TensorBoard 학습 곡선 (36 experiments)
