# RL vs PD Comparison Analysis

## 핵심 비교표

| 접근법 | Dock% | Grasp% | Crash% | 학습 시간 | 환경 |
|--------|-------|--------|--------|----------|------|
| Pure RL v3 (kinematic) | 38.5% | - | - | ~6h | kinematic, 40s |
| Pure RL v3 (dynamic) | 11.4% | 10.0% | - | - | dynamic, 40s |
| Pure RL Phase1 v2 (best) | 24.8% | 19.4% | - | ~3h | dynamic, 3s, box-above |
| Pure RL Phase1 v4 | 23.3% | 18.6% | 3% | ~3h | dynamic, 3s, box-above |
| Pure RL Phase1 v5 | 21.4% | 11.2% | 6% | ~3h | dynamic, 3s, box-above |
| **PD initial (Kp=6)** | **24.7%** | - | **0%** | **0** | **dynamic, 8s** |
| **PD optimized** | **63.8%** | - | **0%** | **0** | **dynamic, 12s** |
| RL residual on PD | <6% | - | 0% | ~1h | dynamic, 12s |

## 핵심 발견

### 1. PD 초기 버전이 이미 RL Best와 동등
PD (Kp=6, Kd=4.5) = 24.7% dock. RL Phase1 v2 best = 24.8%.
**게인 하나도 튜닝하지 않은 PD가 수일간의 RL 학습과 동일한 성능.**

### 2. PD 최적화가 RL을 크게 초월
PD 24.7% → 63.8% (게인 튜닝 + stuck retry + eval 수정).
RL best = 24.8%. **PD가 RL의 2.6배.**

### 3. RL이 PD를 오히려 방해
PD(49%) 위에 RL residual(scale=0.1) 추가 → dock 지표 전부 하락.
RL이 hovering exploit를 발견: dock 시도 안 하고 intermediate reward 수집.

### 4. Crash Rate 차이
PD: 0% crash (모든 실험에서).
RL Phase1 v4: 3%, v5: 6%.
PD는 구조적으로 안전 (acceleration clamp, safety push).

## 왜 RL이 PD보다 못했나

### A. Dynamic Box의 Contact Problem
RL은 "박스에 접촉 → 밀림 → 실패"를 경험해도 credit assignment가 안 됨.
접촉 순간의 미세한 XY 오차(3.7cm vs 6.8cm)가 성공/실패를 결정하지만,
이 차이에서 오는 보상 signal이 너무 약함 (r_approach: step당 0.09 차이).

### B. Reward Shaping Dilemma
- Sparse reward만 쓰면: 225 step 유지 성공이 너무 rare → 학습 불가
- Dense reward 추가하면: hovering exploit 발생 → dock 안 하는 게 최적
- **PD는 이 딜레마가 없음**: 보상 없이 직접 제어

### C. Exploration 비효율
PPO random exploration: step별 random action이 상쇄 → 30 step 누적 효과 없음.
"어느 방향으로든 일관되게 밀면 좋다"를 발견하기 전에 수억 step 소모.
PD: 첫 step부터 올바른 방향으로 이동.

### D. Curriculum Transfer 실패
Phase 1 (box 위 30cm) → Phase 2 (1.5m): 모든 시도 실패.
Phase 1에서 "정밀 안착"만 배우면서 "비행" 능력을 잃음.
PD: 모든 거리에서 동일한 제어법 적용 (gain scheduling으로 자연스러운 전환).

## 왜 PD가 이 Task에 적합한가

### 문제의 본질: 기하학적 제어
"8cm 박스를 10cm opening에 넣기" — 정확한 수학으로 계산 가능.
필요한 건 "학습"이 아니라 "정밀한 position tracking".

### PD의 장점
1. **즉시 동작**: 학습 시간 0, 첫 step부터 유효
2. **투명한 디버깅**: 실패 원인이 게인 값에서 바로 보임
3. **안정성 보장**: acceleration clamp + safety push → 0% crash
4. **Asymmetric design**: 기하학적 마진(X 1cm, Y 6.4cm)을 직접 반영
5. **Stuck recovery**: edge-contact freeze를 감지하고 재시도

### PD의 한계
1. **61-64%가 ceiling**: 같은 변수를 튜닝해도 한 쪽 개선 → 다른 쪽 악화
2. **Contact dynamics**: rigid body 접촉력을 PD로 제어 불가
3. **Wind 적응**: reactive만 가능 (proactive 불가)
4. **Domain shift**: motor kf, mass 변화에 대한 online 적응 없음

## 데이터 파일
- `../data_csv/pd_tuning_progression.csv`: PD 튜닝 전체 과정
- `../data_csv/rl_eval_results.csv`: RL 실험 결과
- `../data_csv/rl_training_summary.csv`: RL TB 메트릭
