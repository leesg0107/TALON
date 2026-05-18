# 03 — Analytical PID Dock 설계 (슬라이드 8-9) ⭐ 핵심

> **C1→C2 transition.** RL 4가지 실패 원인 → 각각에 대응하는 analytical design 1:1 매핑.

## 슬라이드 8: Failure-to-Design Traceability

| RL 실패 원인 (§02) | Analytical Design 대응 |
|---|---|
| Reward dilemma → hover exploit | **Descent gate**: XY 정렬 → 강제 하강 (게임할 reward 없음) |
| Contact velocity 7× overshoot | **Two-stage sigmoid descent**: 0.40→0.15 m/s |
| Curriculum transfer collapse | **Distance-adaptive gains**: approach/precision mode 분리 |
| Box pushed out by contact | **Dock proximity softening**: Kp↓, Kd↑ at contact (compliance proxy) |

## 슬라이드 9: PID 핵심 수식 ([drone_env.py](../../envs/drone_env.py))

### XY: Asymmetric PD (clearance 비대칭 반영)
```python
Kp_x = 12.0, Kd_x = 8.0    # X 축 (strut, 1cm clearance) — aggressive
Kp_y = 8.0,  Kd_y = 6.0    # Y 축 (plate, 6.4cm) — relaxed

dock_proximity = sigmoid((0.04 - alt_above)/0.015) * sigmoid((0.03 - xy_mag)/0.01)
Kp_x -= 2.0 * dock_proximity   # contact 시 12→10
Kd_x += 1.0 * dock_proximity   # damping ↑
```

### Z: XY-gated standoff with PID
```python
xy_coarse = sigmoid((0.10 - xy_mag) / 0.05)
xy_fine   = sigmoid((0.03 - xy_mag) / 0.01)
standoff  = 0.30 - 0.18 * xy_coarse - 0.10 * xy_fine
target_z  = box_z + standoff

# PID: Kp_z=12, Ki_z=3, Kd_z=7 (Ki: mass mismatch 보상)
```

| XY error | Standoff | 동작 |
|---|---|---|
| >15cm | 30cm | high hold (XY 먼저) |
| ~5cm | 12cm | begin descent |
| <2cm | 2cm | final descent |

### Stuck recovery + Safety floor
100-step stall detect → 150-step recovery (lift 30cm above box).
`az += 3.0 * (alt_above < -0.02)` — collision 강제 lift.

## 슬라이드 10: Tuning Progression

**시각**: `plots/pd_tuning_progression.png` (11 PD config → dock_pct 진화)

PD initial (24.7%) → 11회 tuning iteration → **PID optimized 63.8% standalone, ~97% in pipeline**.

## 파일

- `pd_tuning_progression.csv` — 11개 config 변화 (Kp, Kd, descent gate, stuck retry 등 + 결과)
- `plots/pd_tuning_progression.png` — dock% / crash% 진화 chart
- Raw eval JSONs (PD kinematic + dynamic) — `20260415_*_PD_*.json/.txt`
- `system_and_controller_full.md` — system 구조 + PID 전체 수식

## 발표 talking points

1. *왜 4 원인에 1:1 매핑인가?* — design이 *경험적 후처리*가 아니라 *실패 분석으로부터 도출*. paper에서 가장 강한 argument.
2. *Asymmetric Kp_x vs Kp_y*: 기하 (1cm vs 6.4cm)가 *제어 설계에 직접 반영*. Novel-ish.
3. *Two-stage sigmoid descent*: contact velocity 문제를 *gating으로 해결*. 단순하지만 효과적.
4. *결과 24.7% → 63.8%* (standalone) → 11회 tuning → ~97% in pipeline (다른 phase의 wedging).
