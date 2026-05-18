# Thesis Data: Autonomous Aerial Pick-and-Place with Dual-Purpose Landing Gear

## Contribution

**Landing gear를 gripper로 겸용**하는 드론이 물체를 자율적으로 파지하고 운반하는 end-to-end 파이프라인.
RL waypoint navigation + PID precision docking의 **task decomposition** 접근.

```
[Approach]        [Dock]           [Grasp]      [Transport]      [Delivery]
RL waypoint  →  PID analytical  →  Auto-close  →  RL loaded  →  RL waypoint
(Stage 1)        (Stage 3)                        (Stage 4)
```

### Key Results

| 접근법 | Dock% | Crash% | 학습 시간 |
|--------|-------|--------|----------|
| RL best (PPO, dynamic) | 24.8% | ~3% | ~3h |
| PID initial (no tuning) | 24.7% | 0% | 0 |
| PID optimized | **63.8%** | **0%** | 0 |

### Environment
Isaac Lab (Isaac Sim 4.5) · gripper-drone 1.080 kg · 8cm cube 0.2 kg · PPO (SKRL) 4096 envs · 150/300 Hz

---

## Documents

```
thesis_data/
├── README.md                    ← 이 파일
├── 01_system_and_controller.md  ← 시스템 구조 + PID 도킹 제어기 수학
├── 02_rl_training.md            ← Stage 1/4 RL (보상 수식, PPO, 네트워크)
├── 03_docking_experiments.md    ← RL 실험 + RL vs Analytical 비교 + Hybrid 시도
├── 04_end_to_end_pipeline.md    ← 미션 phase state machine
├── raw/
│   └── training_log_original.md ← 원본 학습 로그 (시간순 raw)
└── data_csv/                    ← 그래프/테이블용 CSV + JSON
```

### 논문 섹션 매핑

| 문서 | 논문 섹션 |
|------|----------|
| `01_system_and_controller` | System Design · Methodology · Controller Design |
| `02_rl_training` | RL Methodology · Reward Design · Training Setup |
| `03_docking_experiments` | Experiments · Results · Discussion |
| `04_end_to_end_pipeline` | System Integration · Mission Design |
