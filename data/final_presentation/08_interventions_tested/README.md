# 08 — All Interventions Tested: Ablation (슬라이드 19-20) ⭐ Paper-worthy negative results

## 메시지
**11개 개입을 체계적으로 시도**. 1개만 성공 → "*Phase transition state validation*"이 *fundamental insight*임을 입증.

## 슬라이드 19: 전체 ablation chart

**시각**: `plots/intervention_comparison.png` — 11개 개입 bar chart, baseline 68.4% 기준선

| # | Intervention | End-to-end | Δ vs baseline | Category |
|---|---|---|---|---|
| 0 | **Baseline (loaded_best)** | **68.4%** | — | Original |
| 1 | loaded_grasp_states model | 70.3% | +1.9% | Variant (variance) |
| 2 | **Action clip 1.0** (zero-shot) | **32.5%** | **−36%** | **Failed** ✗ |
| 3 | Action clip 2.0 (zero-shot) | 48.9% | −19% | Failed ✗ |
| 4 | **Aggressive barrier v1** (retrain) | **43.2%** | **−25%** | **Failed** ✗ |
| 5 | Soft tilt-gated barrier v2 | 66.7% | −1.7% | Neutral |
| 6 | Tilt-limiter 45° (controller) | 65% | −3% | Neutral |
| 7 | **Fix 1 (DOCK→CLIMB stability gate)** | **74.4%** | **+6%** ✓ | **Works** |
| 8 | Fix 1+2 (+ HOVER_STAB) | 74.4% | +6% | Same as Fix1 |
| 9 | Fix 1+2+3 (+ lateral damp) | 73.4% | +5% | Neutral |
| 10 | **Fix 1+1.5+2 (+ depth-gate retry)** | **79.6%** | **+11%** ✓ | **Best (with side effect)** |
| 11 | Fix 2.0a (dock_hold_z +2cm) | 45% | −23% | Regressed ✗ |

## 슬라이드 20: 핵심 lesson

**5개 *학습/제어 level* 개입 실패 + 1개 *state validation level* 개입 성공**.

> Paper에서 가장 강한 메시지: "*Hierarchical autonomous pipeline의 fragility는 component-level이 아니라 composition-level. Phase transition state validation은 critical-but-overlooked design point.*"

### 왜 실패했나 (각 category별)

**Action constraints (clip, barrier)**: policy-controller co-adaptation
- 정책이 attitude controller saturation regime에서 작동하게 학습됨
- Zero-shot constraint → 학습된 dynamics 분포와 mismatch → catastrophic
- 학습 시 constraint 넣어도 (barrier retrain): 정책이 hovering exploit 또는 conservative collapse

**Controller-level (tilt-limiter, lateral damping)**: 부작용 또는 무효
- Tilt-limiter: tilt가 binding constraint 아님 (실패는 그 이전에 발생)
- Lateral damping: dock-exit lateral momentum이 root cause 아님 (Q3 데이터로 입증)

**Dock altitude (dock_hold_z + 2cm)**: 잘못된 방향
- box_offset_y가 "depth"가 아니라 "lateral wedging"이었음
- altitude 올리니 plates가 box 위에서 닫혀 wedging 약화 → 더 shallow

**Phase transition validation (Fix 1)**: 성공
- DOCK→CLIMB 진입 시 tilt<15° + ang_vel<2 검증 → climb_failed 48→23
- 깔끔하게 정량적 mechanism 입증 (entry tilt 52° vs 10°)

## 파일
- `intervention_comparison.csv` — 모든 개입 + 카테고리 + 실패 분포
- `plots/intervention_comparison.png` — bar chart with color-coded categories

## 발표 talking points

1. *왜 11개나 시도했나?* — 각 개입은 합리적 가설에서 출발. 실패는 *체계적 분석*에서 옴.
2. *Negative results의 가치*: paper에서 unifying claim ("component-level interventions don't fix composition-level issues")의 evidence
3. *Fix 1만 성공한 의미*: state validation gate는 *engineering trick*이 아니라 *fundamental architectural insight*
4. *Fix 1.5의 trade-off*: retry로 +5% 추가, but box_fell 부작용 (paper에서 limitation으로 명시)
