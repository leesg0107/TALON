# 04 — Climb Phase (슬라이드 11)

## 메시지
Dock 종료 시 drone은 *ground 위 ~0.66m*에 있음. 즉시 delivery 시작하면 박스 들고 ground 근처에서 복잡한 maneuver → 위험. **Climb은 안전 고도(1.0-1.5m)까지 분석적 PID로 올리는 phase.**

## 왜 RL이 아니고 PID인가
- Pure vertical motion (lateral nav 불필요)
- 페이로드 추가됨 → 분석적이면 mass compensation 명시적 가능
- 학습 비용 추가 X

## 왜 *low-gain* PID인가
[drone_env.py:222-228](../../envs/drone_env.py#L222-L228):
```python
Kp_climb = 6.0   # vs dock Kp_z = 12
Kd_climb = 5.0   # vs dock Kd_z = 7
climb_az = climb_az.clamp(min=-2.0)   # thrust cutoff free-fall 방지
```

- 높은 Kp → 빠른 가속 → 페이로드 oscillation → grip 풀림
- Kp=6 → smooth 0.3-0.4 m/s ascent → 페이로드 stable
- min clamp → mass bias로 fast vel 시 huge damping → thrust=0 → free fall 방지

## 슬라이드 시각 자료
`q1_climb_trajectories.png` — climb_fail 3개 + success 1개 trajectory 비교
- 실패 케이스: tilt 50°+ 발산하면서 고도 하락 → ground crash
- 성공 케이스: tilt ~5° 유지, 고도 부드럽게 1.0m 상승

`q1_entry_predictors.png` — DOCK→CLIMB 진입 시 tilt/ang_vel 분포
- **smoking gun**: climb_fail의 mean entry tilt **52°** vs success **10°**
- **Fix 1의 root cause**: DOCK이 destabilized 상태로 넘김 → CLIMB이 못 회복

## False/Fake grasp 검출 (climb 부수 기능)
```python
# False grasp (DOCK 중):
contain_hold ≥ 325 AND box_drone_dist > 0.30 → 그립 실패로 간주, retry

# Fake grasp (CLIMB 중):
gripper_z > pedestal_z + 0.25 AND box_z < pedestal_z + 0.05
→ drone만 올라가고 박스는 그대로 → DOCK으로 retry
```

## 성능
- **Climb→Delivery: ~89-94%** (Fix 1 적용 후)
- 실패 mode: drone+박스 ground crash (10%)

## 발표 talking points
1. *왜 1.5m가 climb target?* — 안전 buffer + delivery 정책의 학습 분포 (z 1.5-3m)
2. *Low-gain의 trade-off*: stable but slow (~3-4초 climb 소요)
3. *Fake-grasp detection*: simulation-only이지만 실세계에서도 동일 메커니즘 가능
