# 09 — Limitations & Future Work (슬라이드 21-22)

## 슬라이드 21: 한계 (정직하게)

### Scope 한계
- **Sim-only** — Isaac Lab PhysX, sim-to-real 미수행
- **Single object** — 8cm cube only (shape/size generalization 없음)
- **No external perception** — box position은 ground-truth (vision/state estimation 없음)

### Performance 한계
- **End-to-end 74-80%** — top-tier 기준(90%+)에 못 미침
- **Dock의 stochastic Z variance** — 미해결
  - 외부 변수 (yaw, mass, wind, plate sync) 모두 음성 (Cohen's d, correlation 모두 약함)
  - PhysX 수치 noise / 정책-제어기 coupling에서 발생 추정
- **box_dropped + too_tilted_delivery ~ 9-12% 잔여** — dock 품질에 sensitivity

### Methodological 한계
- **Fix 1+1.5+2의 +11.2%가 baseline variance(±3-5%) 안에 있을 수 있음** → 통계적 significance 부족 가능 (multi-seed × multi-run 필요)
- **6개 component-level intervention 실패의 unifying theory 부재** — paper에서 "engineering observation"으로만 framing 가능

## 슬라이드 22: 향후 작업

### Short-term (~1-2 months)
1. **Multi-seed × multi-run statistical analysis** — 결과 statistical significance 확보
2. **Pre-close settle phase** — plate close 전 0.5초 z-settle (drone_z variance ↓ 시도)
3. **Z PID gain tuning** (Kp 12→16, Kd 7→10) — variance ↓ 시도

### Mid-term (~3-6 months)
4. **Multiple object types** — 6cm / 10cm / non-cube shapes
5. **Vision-based box detection** — state estimation noise 현실화
6. **Sim-to-real prototype** — physical drone + 1cm-clearance gripper

### Long-term (paper extension)
7. **Anisotropic safety constraint formal analysis** — 1cm × 6.4cm clearance의 control-theoretic 표현 (안전 envelope 유도)
8. **Phase transition validation general framework** — 다른 hierarchical pipeline (manipulation, navigation)에 적용

## 발표 결론 talking points

1. *솔직히*: 본 연구는 *workshop / thesis chapter 수준*, top-tier conference에는 더 polish 필요
2. *Contribution이 무엇인가?*: (a) failure-to-design 1:1 매핑 PID, (b) diagnostic-driven methodology, (c) phase transition validation insight
3. *왜 paper-worthy인가?*: 단순 demo가 아니라 *왜 작동하고/안 하는지* 정량 입증
4. *어디로 갈 것인가?*: 본 thesis는 *foundation*, 다음 단계는 sim-to-real + multi-object
