# 발표용 자료 인덱스

발표 흐름에 따라 순서대로 정리. 각 폴더 README가 슬라이드별 내용 가이드.

## 슬라이드 흐름 (권장 20-25분 발표)

| # | 폴더 | 슬라이드 내용 | 핵심 자료 |
|---|---|---|---|
| 1 | `00_overview/` | 동기 + 시스템 개요 + 데모 영상 | `demo_end_to_end.mp4`, `architecture.md` |
| 2 | `01_approach_RL/` | Stage 1 RL waypoint 학습 | `waypoint_eval_summary.png`, eval JSONs |
| 3 | `02_dock_RL_failure/` | **Dock RL 시도 + 실패 4 원인** | `stage3_learning_curves.png`, `rl_eval_comparison.png`, `docking_experiments_full.md` |
| 4 | `03_dock_PID_solution/` | **Analytical PID 설계** | `pd_tuning_progression.png`, `system_and_controller_full.md` |
| 5 | `04_climb/` | Climb phase 필요성 + low-gain PID | `q1_climb_trajectories.png` |
| 6 | `05_delivery_RL/` | Stage 4 loaded RL + grasp_states pipeline | `delivery_design.md` |
| 7 | `06_end_to_end_results/` | 통합 결과 (waterfall + failure breakdown) | `phase_success_waterfall.png`, `failure_breakdown.png` |
| 8 | `07_failure_analysis/` | 정량 진단 (v1, v2, v3) → mechanism 규명 | 12개 진단 plot + raw npz |
| 9 | `08_interventions_tested/` | **11개 개입 ablation** (paper의 부정적 결과 narrative) | `intervention_comparison.png` |
| 10 | `09_limitations_future/` | 한계 + 향후 작업 | `README.md` |

## 발표 길이별 추천 깊이

| 시간 | 강조 슬라이드 | 생략 가능 |
|---|---|---|
| 10분 | 1, 3, 4, 7 | 5, 6, 8, 9, 10 |
| 20분 | 1-9 모두 (각 1-2 슬라이드) | — |
| 30분+ | 1-10 모두 + 8번 상세 ablation 분석 | — |

## 핵심 수치 (cheat sheet)

```
Approach→Dock:   100% (Stage 1 RL flight)
Dock→Climb:      ~97% (PID, was 24.8% in RL attempt)
Climb→Delivery:  ~90% (PID low-gain)
Delivery→Done:   ~78% (Stage 4 RL loaded)
End-to-end:      ~68-74% baseline → 79.6% peak with Fix 1+1.5+2

Best RL dock:    24.8% (dynamic box, Stage 3 PPO)
PID initial:     24.7%
PID optimized:   63.8% standalone → 97% in pipeline
```

## 파일 형식

- `.md`: 슬라이드 narrative / 참고 텍스트
- `.png`: 그래프 (슬라이드에 직접 삽입)
- `.csv`: 원시 수치 (필요 시 다른 plot 생성용)
- `.json`: raw eval 결과
- `.npz`: numpy 진단 데이터 (offline 재분석용)
- `.mp4`: 데모 비디오
