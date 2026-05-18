# 01 — Stage 1: Approach RL (슬라이드 4)

## 메시지
드론이 박스 위 0.5m 지점까지 *waypoint를 따라가는* RL 정책. PPO로 ~1B step 학습. **성공률 100%** (eval).

## 학습 설계
- **Obs 22D**: vel_b(3) + ang_vel_b(3) + R_flat(9) + goal_b(3) + prev_action_norm(4)
- **Action 4D**: `[ax, ay, az, yaw_ref]` (body frame, scaled to [-8,8] m/s² + [-π,π])
- **PPO (SKRL)**: 4096 parallel envs, 2×128 MLP ELU, lr 3e-4, γ=0.99, λ=0.95
- **Domain Randomization**: mass ±10%, motor thrust ±15%, wind 0.5N sinusoidal, sensor noise

## 핵심 reward (gripper_waypoint_env.py:209-218)
```python
r_direction = 0.5 * clamp(vel_toward_goal, -3, 3)       # 진행 방향 속도
r_arrive = arrived * 10.0 / (time_sec + 0.5)            # time-decayed bonus
r_crash = -5.0 * (altitude < 0.15)                       # 안전
r_smooth = -0.01 * sum(Δaction²)                         # 부드러움
r_tilt = -3.0 * clamp(tilt - 30°, min=0)                # tilt threshold
r_timeout = -2.0 * (steps_since_goal > 450)             # 3s/WP
```

## 파일
- `waypoint_eval_flight_*.json` (4개) — 학습 후 평가 결과
- `waypoint_eval_summary.csv` — JSON 컴파일
- `waypoint_eval_summary.png` — bar chart
- `rl_training_full.md` — 상세 학습 문서 (thesis_data에서)

## 발표 talking points
1. *왜 RL?* — wind, mass 변동에 적응성. analytical waypoint follower도 가능하지만 robustness ↓
2. *왜 22D?* — gripper-centric obs design (state representation 일관성)
3. *결과 100%* — 이건 *쉬운 phase*. 진짜 문제는 dock에서 시작.
