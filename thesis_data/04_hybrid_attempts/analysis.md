# Hybrid Controller Attempts (PD + RL Residual)

## 시도 1: Fixed Residual Scale 0.3 + 기존 보상

**설정**: PD analytical base + RL output * 0.3
**문제**: RL output=0이 최적. PD가 모든 approach를 담당하고 r_approach가 올라감.
RL은 "아무것도 안 해도 보상이 오는" 상태.
**결과**: 구현만 하고 학습 미진행 (분석 단계에서 문제 발견)

## 시도 2: Outcome-Based 보상 + Adaptive Scale

**설계 원칙**: 
- PD가 하는 일(approach)의 보상 축소
- RL이 할 수 있는 일(precision, soft contact)의 보상 추가
- Adaptive scale: 0.5 (far) → 0.15 (near)

**비판적 검토에서 발견된 문제**:
- Adaptive scale에서도 near=0.15일 때 RL(±1.2 m/s²)이 PD(0.4 m/s²)의 3배 → PD dominant 아님
- 고정 0.1로 변경

## 시도 3: Fixed Scale 0.1 + Outcome 보상 (실제 학습)

**보상 구조**:
```python
# 축소 (PD 담당)
r_approach_xy: 3.0 → 1.0
r_approach_z:  2.0 → 0.5
r_fine: 삭제

# 유지 (dock 결과)
r_contain, r_dock_bonus, r_hold_stable, r_success: 그대로

# 추가 (RL 정밀도)
r_descent_precision = 5.0 * exp(-30 * xy_err) * is_descending
r_soft_contact = 3.0 * near_contact * exp(-3|vz|) * is_descending  # hover exploit 방지
r_x_precision = 4.0 * exp(-60 * |box_local_x|)

# 축소 (residual용)
r_smooth: -0.2 → -0.1
r_mag: -0.1 → -0.02
```

**Hover exploit 방지**: r_soft_contact에 `is_descending` gate 추가.
v5의 r_hover_dock 실패 교훈: "박스 근처에서 느리면 보상" → hovering exploit.
수정: "박스 근처에서 **하강 중에** 느리면 보상" → hovering은 보상 0.

**학습 결과 (5% 시점)**:
| 지표 | 시작 (PD only) | 5% 학습 후 | 변화 |
|------|---------------|-----------|------|
| overlap | 0.042 | 0.004 | **↓10x** |
| r_success | 0.009 | 0.001 | **↓9x** |
| r_contain | 0.999 | 0.094 | **↓10x** |
| Total reward | 3811 | 5163 | ↑ (나쁜 징조) |

**실패 원인**: Total reward 상승 + dock 지표 하락 = hovering exploit.
r_soft_contact에 is_descending 붙였지만, r_approach(1.0) + r_x_precision(4.0) + r_column(6.5) = 11.5/step을 hovering으로 수집 가능.
**dock 시도의 risk (box_fell terminate) > hovering의 안전한 11.5/step.**

## 시도 4: INDI Outer Loop

**동기**: Motor kf ±15%로 인한 steady-state error 보상
**결과**: 완전 발산 (dock 6% → 0.1%)

| 설정 | Dock% | v_z | Tilt |
|------|-------|-----|------|
| α=0.3, gain=1.0 | 6.0% | +0.3 (상승!) | 18-25° |
| α=0.1, gain=0.3 | 0.1% | +0.5 (상승!) | 27-30° |

**원인**: Outer loop INDI ↔ inner loop attitude controller 충돌.
Attitude controller 응답 지연(~13ms) > INDI 보정 주기(6.7ms).
INDI가 "아직 반응 안 했으니 더 보정" → 양성 피드백 → 발산.

## 시도 5: PID (Conditional Integral)

| 설정 | Dock% | 변화 |
|------|-------|------|
| Ki=2.0, global | 55.8% | -2.1% |
| Ki=1.0, global | 53.2% | -4.7% |
| Ki=1.5, xy<20cm only | 54.7% | -3.2% |

**모두 PD only(57.9%)보다 나빠짐.**
**원인**: approach 중 integral 누적 → 도착 시 overshoot.
Conditional (xy<20cm에서만)도 효과 없음: 20cm 경계에서 integral on/off가 불안정.

## 핵심 교훈

1. **RL residual은 reward 설계가 극도로 어려움**: dock 안 하는 게 거의 항상 더 높은 보상
2. **INDI는 outer loop에서 구조적으로 안 맞음**: inner loop 지연과 충돌
3. **PID의 integral은 approach task에 역효과**: transient 오류에 integral 무용
4. **PD only가 이 task의 sweet spot**: 복잡도 대비 성능 최적

## 데이터 파일
- `../data_csv/rl_training_summary.csv`: stage3_residual TB 데이터 포함
