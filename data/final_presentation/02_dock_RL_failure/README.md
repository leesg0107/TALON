# 02 — Dock RL 시도 + 실패 분석 (슬라이드 5-7) ⭐ 핵심

> **이 섹션이 발표의 main contribution.** RL이 왜 안 됐는지 정량 입증 → 다음 섹션 PID 설계 정당화.

## 슬라이드 5: Stage 3 RL Setup
- **Obs 31D**: gripper-centric — body vel, ang vel, R, goal_b, goal_world, plate_angles, box_in_gripper, grasp_flag, payload, box_dist
- **Action 8D**: body accel(3) + body rate(3) + yaw + gripper command
- **Reward**: overlap-based (15.0 * overlap_ratio + 10.0 * full_contain + r_x_align + r_y_align + r_z + r_success 50.0)
- **PPO Stage 3**: 4096 envs, 2×128 MLP

## 슬라이드 6: 실패 결과
**시각**: `plots/stage3_learning_curves.png` (3 panels — Total reward, overlap_ratio, xy_err)

핵심 시리즈:
- `learning_curve_stage3_phase1_v4.csv` — Phase1 v4 (TB overlap 0.422 → eval 23%)
- `learning_curve_stage3_dynamic_v2.csv` — Dynamic v2 (11% on dynamic box)
- `learning_curve_stage3_safety_v1.csv` — 최종 시도 (safety reward)

**시각**: `plots/rl_eval_comparison.png` — 모든 RL 실험 dock% vs crash% bar chart

핵심 표 (`rl_eval_results.csv`):
| Model | Env | Dock% |
|---|---|---|
| Stage 3 v2 | kinematic | 31.2% |
| Stage 3 v3 | kinematic | 38.5% |
| Stage 3 v3 | **dynamic** | **11.4%** ← 폭락 |
| Phase 1 v2 (best PPO) | dynamic | **24.8%** |
| **PID (initial)** | dynamic | **24.7%** |
| **PID (optimized)** | dynamic | **63.8%** |

## 슬라이드 7: 4 가지 실패 원인 (root cause)

상세는 `docking_experiments_full.md §B` 참조.

### 원인 1: Dynamic box contact dynamics
- Kinematic 38.5% → Dynamic 11.4%
- 접촉 순간 미세한 XY 오차(3.7cm → 6.8cm)가 성공/실패 결정
- Reward signal이 너무 약함 → credit assignment 실패

### 원인 2: Reward shaping dilemma
- **Sparse (r_success만)**: 225 step contain 너무 rare → 학습 안 됨
- **Dense (r_approach 등)**: hovering exploit → "dock 안 하는 게 최적"
- 어떤 intermediate reward도 hovering이 더 높은 가치

### 원인 3: Exploration 비효율  
- PPO random action은 step별로 상쇄 → 누적 효과 없음
- Residual scale 0.1: step당 0.02mm 변화 → noise 수준

### 원인 4: Curriculum transfer 구조적 한계
- Phase 1 모델 (30cm 범위) → Phase 2 (1.5m) 전환 시 전수 실패
- Analytical은 모든 거리에서 동일 법칙으로 자연 전환

## 파일

### CSV (10 learning curves + 5 summary)
- `learning_curve_stage3_*.csv` — Stage 3 학습 곡선들
- `learning_curve_8s_sigmoid.csv`, `learning_curve_dynamic_v*.csv`, `learning_curve_safety_v*.csv`
- `rl_eval_results.csv` — 19개 RL 모델 평가 (dock_pct, grasp_pct, crash_pct 등)
- `reward_design_comparison.csv` — reward 설계별 peak metrics
- `rl_training_summary.csv` — 학습 메트릭 start/end/max
- `dock_vs_overlap_lost_at_first_overlap.csv` — 실패 모드 정량

### Raw eval JSONs
- Stage 3 safety_v1 eval (RL 최종 시도 데이터)

### Plots (생성됨)
- `plots/stage3_learning_curves.png` — 3-panel 학습 곡선
- `plots/rl_eval_comparison.png` — 모든 실험 비교 bar chart

### 문서
- `docking_experiments_full.md` — 전체 RL 도킹 실험 문서 (Part A-D)
- `training_log_original.md` — 시간순 학습 로그
