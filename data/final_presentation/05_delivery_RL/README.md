# 05 — Stage 4: Delivery RL with Payload (슬라이드 12)

## 메시지
페이로드를 들고 delivery target까지 운반. **Stage 1과 같은 PPO 구조**, 단지 **23D obs** (+payload mass) + **tighter tilt threshold** + **overshoot penalty**.

## 학습 차이 (vs Stage 1)
| | Stage 1 (Approach) | **Stage 4 (Delivery)** |
|---|---|---|
| Obs dim | 22D | **23D** (+ payload_mass est) |
| Action dim | 4D | 4D |
| Gripper | open | closed |
| Tilt threshold | 30° (r_tilt 활성) | **26°** (페이로드 보호) |
| Overshoot penalty | — | **r_overshoot** (1.5m/s 이상 + near WP) |
| Box | flying without | **pre-attached in gripper** |
| Initial state | random spawn | sampled from grasp_states.pt |

## 학습 pipeline (multi-stage transfer)

```
Stage 1 (flight, from scratch)
    ↓ warm-start (22D → 23D, last column zero-init)
    ↓ --reset_std (exploration 다시 열기)
Stage 4 Base (loaded, idealized box placement)
    ↓ checkpoint
Stage 4 Grasp (loaded, physical grasp states 사용)
```

## Pre-simulated Grasp States ([data/grasp_states.pt](../../data/grasp_states.pt))
- ~5,000 physical grasp 상태 사전 수집
- [scripts/generate_grasp_states.py](../../scripts/generate_grasp_states.py)로 analytical dock controller를 반복 실행
- 매 Stage 4 reset 시 random sample → realistic grasp 분포로 학습

**Keys**: `drone_pos_local`, `drone_quat`, `drone_vel`, `box_pos_local`, `box_quat`, `joint_pos`, `mass_scale`, `payload_mass`, `motor_kf_scale`, `n_states`

## Reward 차이 ([gripper_waypoint_env.py:233-265](../../envs/gripper_waypoint_env.py))
```python
# Stage 1 + Stage 4 공통:
r_direction + r_arrive + r_crash + r_smooth + r_angular + r_tilt - 2.0*timeout

# Stage 4 추가:
r_overshoot = -1.0 * near_wp * clamp(speed - 1.5, min=0)
# → near WP에서 1.5 m/s 초과 시 페널티 (U-turn 방지)
```

## 성능
- **Delivery→Done: ~78-92%** (Fix 1+1.5+2 적용 후)
- 실패 mode: 
  - box_dropped_delivery (~10%): 박스가 gripper에서 미끄러져 떨어짐
  - too_tilted_delivery (~8%): drone tilt > 70° 발산

## 발표 talking points
1. *왜 Stage 1 → Stage 4 warm-start?* — 학습 비용 절약 + flight skill 유지
2. *왜 tilt threshold 26° (Stage 1은 30°)?* — 페이로드 CoM shift 막기
3. *왜 grasp_states.pt 사용?* — 학습 분포 ↔ 추론 분포 align (real physical grasp)
4. *주요 한계*: dock 단계의 grasp 품질 variance → delivery 분포 outlier → 실패 (§07-08에서 상세)

## 파일
- (별도 CSV/plot 없음 — Stage 4 학습 곡선은 archive/old_logs/runs에 TensorBoard로 존재)
- `data/grasp_states.pt` 참조
