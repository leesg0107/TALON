# 00 — Overview & Motivation (슬라이드 1-3)

## 슬라이드 1: 타이틀 + 동기
**제목 후보**: "Hybrid RL-Analytical Aerial Manipulation with Dual-Purpose Landing Gear"

**메시지**: 드론의 *landing gear를 gripper로 겸용*하여 추가 매니퓰레이터 없이 자율 파지+운반.

**시각**: `demo_end_to_end.mp4` — 전체 mission 자율 수행 데모 (15-30초 cut)

## 슬라이드 2: 시스템 구조
**메시지**: 4-phase pipeline.

```
[Approach RL]  →  [Dock PID]  →  [Climb PID]  →  [Delivery RL]
   22D obs           analytical      low-gain        23D obs + payload
   4D action         asymmetric PD                   warm-start from flight
   PPO Stage 1       failure-to-design               PPO Stage 4
```

**핵심 design choice**: 
- RL = navigation (적응성, wind 대응)
- Analytical = docking (1cm clearance, 기하적 정밀도)
- 이건 *empirical 결정*: RL이 dock에서 24.8%까지밖에 못 함 → §02에서 상세

## 슬라이드 3: 핵심 결과 미리보기
| 단계 | RL 시도 | Analytical |
|---|---|---|
| Dock 단독 | **24.8%** | **63.8%** standalone, **~97%** in pipeline |
| 학습 시간 | ~3h × 수 회 | 0 |
| Crash rate | ~3% | 0% |

End-to-end: **~74-80%** (Fix 적용 후)

## 파일

- `demo_end_to_end.mp4` — 전체 mission demo 비디오 (10 MB)
- `drone_urdf_reference.urdf` — gripper-drone URDF 설계
- `thesis_data_README.md` — 사전 준비된 thesis 데이터 가이드 (참고용)
