# 06 — End-to-End Results (슬라이드 13-14) ⭐ Headline

## 슬라이드 13: Phase-by-Phase Performance

**시각**: `plots/phase_success_waterfall.png` (좌: bar chart, 우: cumulative line)

| Phase Transition | Baseline | Fix 1+1.5+2 |
|---|---|---|
| Approach → Dock | **100%** | 100% |
| Dock → Climb | 97.4% | ~99% |
| Climb → Delivery | 89.9% | ~94% |
| Delivery → Done | 78.1% | ~92% |
| **End-to-end** | **68.4%** | **~80%** |

Eval condition: 128 parallel envs × 500 missions, DR (mass ±10%, motor ±15%, wind 0.5N, payload 0.15-0.25kg).

## 슬라이드 14: Failure Breakdown

**시각**: `plots/failure_breakdown.png` — stacked bar (Baseline vs Fix 1+2 vs Fix 1+1.5+2)

핵심 failure modes (per 500 missions):

| Failure | Baseline | Fix 1+2 | Fix 1+1.5+2 |
|---|---|---|---|
| climb_failed | 48 | 23 | **17** ↓65% |
| box_dropped_delivery | 44 | 48 | **14** ↓68% |
| too_tilted_delivery | ~52 | ~30 | ~30 |
| dock_timeout (variants) | 11 | 11 | 11 |
| box_fell_during_dock | 5 | 5 | 25 (retry 부작용) |

## 파일
- `pipeline_full.md` — 8-phase state machine 상세 (thesis_data)
- `failure_breakdown_by_config.csv` — config별 정량 비교
- `plots/phase_success_waterfall.png` — 단계별 + 누적 성공률
- `plots/failure_breakdown.png` — failure mode stacked bar

## 발표 talking points

1. *Approach 100%인데 왜 전체 68%?* — composition effect. 각 phase 90%+여도 4개 곱하면 65%
2. *어디가 bottleneck?* — Delivery (78%) > Climb (90%) > Dock (97%)
3. *Fix 적용 후 어디서 효과?* — climb_failed 65% ↓, box_dropped 68% ↓
4. *왜 box_fell_during_dock이 늘었나?* — Fix 1.5 retry mechanism의 부작용 (다음 섹션에서)
