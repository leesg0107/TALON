# Thesis Data: Dual-Purpose Landing Gear Drone Grasping

Isaac Lab 시뮬레이션 환경에서 그리퍼-드론의 물체 파지 실험 데이터 및 분석.
논문 작성을 위한 데이터/분석 아카이브.

## 폴더 구조

```
thesis_data/
├── README.md                          ← 이 파일
├── 01_rl_experiments/
│   ├── analysis.md                    ← RL 실험 전체 분석 (Stage 3)
│   └── training_log_original.md       ← 원본 학습 로그 (시간순, 모든 시도)
├── 02_pd_controller/
│   └── analysis.md                    ← PD 컨트롤러 설계/튜닝/결과 분석
├── 03_rl_vs_pd_comparison/
│   └── analysis.md                    ← RL vs PD 비교 분석 + 왜 PD가 더 나았는가
├── 04_hybrid_attempts/
│   └── analysis.md                    ← PD+RL residual, INDI, PID 시도 및 실패 분석
└── data_csv/
    ├── rl_training_summary.csv        ← TB 메트릭 (8 experiments)
    ├── rl_eval_results.csv            ← RL eval 결과 (14 entries)
    ├── pd_tuning_progression.csv      ← PD 게인 튜닝 전 과정 (12 steps)
    └── dock_vs_overlap_lost_at_first_overlap.csv  ← 성공/실패 접촉 비교
```

## 핵심 결과 요약

| 접근법 | Dock% | Crash% | 학습 시간 |
|--------|-------|--------|----------|
| Pure RL best (Phase1 v2) | 24.8% | ~3% | ~3h |
| PD initial (no tuning) | 24.7% | 0% | 0 |
| PD optimized | **63.8%** | 0% | 0 |
| PD + RL residual | <6% | 0% | 1h (실패) |

## 데이터 활용 가이드

### 그래프 제작용 CSV
- `pd_tuning_progression.csv`: PD 성능 향상 곡선 (12 데이터 포인트)
- `rl_eval_results.csv`: RL 실험별 성능 비교 막대 그래프
- `dock_vs_overlap_lost_at_first_overlap.csv`: 성공/실패 접촉 조건 비교

### 분석 문서 → 논문 섹션 매핑
- `01_rl_experiments/analysis.md` → Related Work / Methodology / RL Results
- `02_pd_controller/analysis.md` → Proposed Method / Controller Design
- `03_rl_vs_pd_comparison/analysis.md` → Discussion / Comparison
- `04_hybrid_attempts/analysis.md` → Discussion / Hybrid Approaches

## 실험 환경
- Simulator: Isaac Lab (Isaac Sim 4.5)
- Robot: Custom gripper-drone (URDF, 1.08kg)
- Box: 8cm cube, 0.2kg, dynamic (gravity ON)
- Pedestal: 30×30×50cm, kinematic
- RL: PPO (SKRL), 4096 parallel envs
- Control rate: 150Hz (policy), 300Hz (physics)
- Domain randomization: mass ±10%, motor kf ±15%, wind 0.5N, pos noise 2cm

## 날짜
- RL 실험: 2026-04-10 ~ 2026-04-11
- PD 개발: 2026-04-14
- Hybrid 시도: 2026-04-14
- 문서 정리: 2026-04-15
