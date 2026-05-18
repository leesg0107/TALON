# 07 — Failure Analysis: Quantitative Diagnostics (슬라이드 15-18) ⭐ Methodology Contribution

## 메시지
**진단 3 단계 (v1 → v2 → v3)**로 failure mechanism을 *경험이 아닌 데이터로* 규명.

## 슬라이드 15: v1 — Delivery 실패는 어디서?

**시각**: `trajectory_comparison.png` (3 fail trajectories + 2 success, tilt + lat_accel)

핵심 발견:
- 실패의 46%가 **ramp** (lat_accel 누적 상승), 54%가 **sustained high**
- **Spike 0%** — action discontinuity는 원인이 아님
- 정책이 **물리적으로 실현 불가능한 lat_accel 명령** (실패 트래젝토리: 평균 89 m/s², 8g+)

**시각**: `histograms.png` — max_tilt, max_lat_accel 분포 (success vs fail)
- Success max_tilt: tight peak ~25-30° (r_tilt threshold 직전)
- Fail max_tilt: bimodal, 70° peak (termination 한계)

## 슬라이드 16: v2 — Climb 실패 + propagation

**시각**: `q1_entry_predictors.png` — DOCK→CLIMB 진입 시 (tilt, ang_vel) 분포
- **Smoking gun**: climb_fail의 mean entry tilt **52°** vs success **10°**
- → DOCK이 destabilized 상태로 넘기는 게 root cause

**시각**: `q1_climb_trajectories.png` — climb 실패 trajectories (tilt 발산, 고도 하락)

**시각**: `q3_dock_vs_delivery.png` — **모든 점이 y=x 위!**
- box_offset_y at DOCK exit = box_offset_y at DELIVERY entry
- **CLIMB은 grasp depth를 전혀 바꾸지 않음** (Δ drift = 0)
- → shallow grasp의 원인은 100% DOCK

**시각**: `q4_dock_trajectory.png` — box_offset_y가 contain 230-280에서 분기
- plate closure 순간이 결정적
- success: -0.028로 wedge / shallow: -0.016에서 멈춤

## 슬라이드 17: Q5 — Mechanism split

**시각**: `q5_mechanism_split.png` — box_dropped vs too_tilted_delivery 분리 분석

| Feature | s vs box_dropped (Cohen d) | s vs too_tilted (Cohen d) |
|---|---|---|
| **box_offset_y @climb_entry** | **-1.07** | **-1.07** ← 동일 |
| vel_z @delivery_entry | +1.40 | +0.91 |
| ang_vel @delivery_entry | -0.29 | **-0.88** ← 차이 |

→ **둘 다 shallow grasp이 1차 원인** (d=-1.07로 동일). 2차 mechanism만 다름 (ang_vel 발산 여부).

## 슬라이드 18: v3 — Dock controller 변동의 origin?

**시각**: `predictive_features.png` + `v3_trajectories.png`

14 feature 분석 결과 — **5 가설 모두 음성**:
| 가설 | corr(feature, final_y) |
|---|---|
| yaw 결정론 | -0.05 |
| plate 비대칭 | std=0 (대칭) |
| drone-box world XY | -0.14 |
| 초기 box velocity | -0.11 |
| mass_scale | -0.01 |

→ **외부 결정 변수 없음**. drone_z의 ±1cm variance가 marker이지만 *대응되는 외부 원인 없음*. **Stochastic Z variance × plate close geometry 민감도**가 진짜 mechanism.

## 파일

### 진단 plots
- `trajectory_comparison.png`, `histograms.png` (v1)
- `q1_climb_trajectories.png`, `q1_entry_predictors.png` (v2 Q1)
- `q2_delivery_entry.png` (v2 Q2)
- `q3_dock_vs_delivery.png`, `q4_dock_trajectory.png`, `q5_mechanism_split.png` (v2 Q3-5)
- `predictive_features.png`, `scatter_predictors.png` (dock mechanism)
- `v3_trajectories.png` (v3 — 9 features over time)

### Raw npz (재분석용)
- `diagnose_delivery_raw.npz` (v1)
- `diagnose_climb_delivery_raw.npz` (v2)
- `diagnose_climb_propagation_raw.npz` (v2 enriched)
- `diagnose_dock_mechanism_raw.npz` (v3 with 14 features)

## 발표 talking points

1. **Diagnostic-driven methodology**가 paper의 가장 *novel한 contribution*
2. **C1 → C2 → C3 인과 chain**: failure 메커니즘이 design choice를 *경험이 아닌 데이터로* 정당화
3. **Stochastic mechanism의 honest 보고** — 모든 외부 변수가 음성 → fundamental limitation
4. **Cohen's d / correlation analysis**: 정량 분석 framework — paper에서 강한 evidence
