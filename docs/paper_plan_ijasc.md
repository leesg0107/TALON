# Paper Plan — IJASC (International Journal of Advanced Smart Convergence)

Drafted 2026-07-02. Framing selected by a 3-lens judge panel (skeptical-reviewer /
venue-fit / evidence-audit) over three candidate framings; "systems + composition
analysis" won unanimously. All numbers below verified against
`logs/graspfix_{off,full,regrasp}_seed1.json`, `logs/diagnose_climb_propagation_raw.npz`,
and `data/final_presentation/{10,11}_*/`. No new sim runs required except the
optional arms listed at the end.

---

## Paper title

**When Every Stage Passes but the Mission Fails: Grasp Depth as a Hidden
Interface Variable in an Autonomous Aerial Pick-and-Deliver System**

(alt: *...: Composition Loss and the Limits of No-Retrain Interventions in a
Multi-Skill Aerial Transport Mission*)

## Main contributions

1. **System (C1).** A complete simulated autonomous aerial pick-and-deliver
   system whose two landing-gear plates double as parallel gripper fingers,
   combining RL waypoint policies with an analytical PID dock/climb controller
   (motivated by RL docking plateauing at 24.8% on dynamic objects) in an
   8-phase mission state machine — 72.2% end-to-end over 500 domain-randomized
   missions with fully physics-based grasping at 10 mm strut clearance.
2. **Diagnosis (C2).** Identification and statistical characterization of grasp
   depth as a *hidden interface variable* that caps the pipeline: invisible to
   the stage success metric (0 of 383 dock accepts rejected for depth despite
   35% shallow), stochastic (pre-close state predicts final depth at R²=0.066),
   irreversible (climb-entry → delivery-entry depth corr 0.9994, mean drift
   0.055 mm), class-specific and orthogonal (depth→delivery failure Fisher
   p=1e-12; tilt→climb failure p=7e-12; cross-terms n.s.), and dominant
   (mission success 79.1% deep vs 53.3% shallow, Spearman ρ=0.30, p=2e-9).
   The same variable resolves the standalone-vs-pipeline paradox as covariate
   shift: the delivery training bank is 99.2% deep while the live pipeline
   feeds 37% shallow; pipeline success on deep grasps (91.1%) matches
   standalone (~88%).
3. **Falsification (C3).** A same-seed evaluation showing all three no-retrain
   intervention families at this interface fail to raise end-to-end success:
   - *Control shaping* (ramped plate close, measured within the shape+gate
     arm): deep-grasp share collapses 47.1%→0.0% at the −25 mm threshold
     (mean depth −22.3→−10.3 mm, MW p≈3e-44) — the snap-close is what seats
     the box; the arm scores 38.0% [33.3, 43.0] with 52.1% gated rejections.
   - *Bounded re-grasp* (detect-shallow-and-redraw, K=3): end-to-end
     69.0%→63.0% (Fisher p=0.16, n.s.) — delivery-phase failures ARE cut
     significantly (18.0%→10.4% of missions, p=0.020) but the retry process
     relocates an equal failure mass to the dock phase (0.3%→16.1%: box-fell
     4.7%, retry timeouts 8.9%, budget exhaustion 2.6%). The analytic
     independent-redraw projection (~78%) is falsified by measurement.
   - **Net finding: risk relocation, not risk removal.**
4. **Interface monitoring value (C4).** Even without raising success, interface
   measurement is cheap and operationally valuable: entry-state signals predict
   failure class-specifically (ROC AUC 0.763–0.836, bootstrap CIs exclude 0.5);
   a tilt gate at 5% false-reject captures 67% of climb failures (p=2.4e-19);
   and the measured shape+gate arm raises success-given-accepted-handoff from
   69.2% to 88.5% (p<1e-4) while cutting delivery-phase failures 18.0%→1.8% of
   missions. Across arms, conditional reliability rises monotonically
   (69.2 → 75.2 → 88.5%) as end-to-end throughput falls (69.0 → 63.0 → 38.0%)
   — a **throughput-vs-certified-reliability operating knob** for mission
   designers (fragile-cargo missions choose the certified end).
5. **Design guidelines (C5).** Data-backed checklist for composing learned and
   analytical skills: instrument continuous interface quality alongside boolean
   stage success; test interface variables for irreversibility before investing
   in downstream fixes; treat interface quality as a first-class
   training-distribution variable (the survivor-filtered grasp bank left
   exactly the failing grasps OOD); budget for rejection/retry rather than
   assuming shaping can move a contact-formed stochastic variable; and measure
   interventions rather than modeling them.

## Main differences from prior research

1. **Funnel/sequential composition** (Burridge, Rizzi & Koditschek IJRR 1999;
   Tedrake et al. LQR-Trees IJRR 2010): these certify composition via boolean
   set-containment at phase boundaries — exactly the predicate our data
   falsifies: the drone satisfies containment at every handoff, yet end-to-end
   success is governed by a continuous quality variable *inside* the accepted
   set that funnel-style checks never inspect.
2. **RL skill chaining / policy sequencing** (Konidaris & Barto NeurIPS 2009;
   T-STAR CoRL 2021; Clegg et al. RA-L 2020; transition policies ICLR 2019):
   these repair composition by reshaping boundary state distributions at
   training time; we show the dominant coupling here is not a state-space
   mismatch the downstream policy rejects but a stochastic, irreversible latent
   scalar it *accepts* — and we quantify empirically what each no-retrain
   remedy does instead (shaping backfires, gating certifies-but-rejects,
   resampling relocates). Our own survivor-filtered training bank is an
   instance of the very problem distribution-matching was meant to solve
   (train ≠ inference at exactly the failing grasps).
3. **Aerial grasping / manipulation** (Fishman et al. IROS 2021; SNAG Science
   Robotics 2021; RAPTOR IROS 2022; surveys RA-L 2018, T-RO 2022): this
   literature engineers hardware/control so the isolated grasp event succeeds
   and reports per-grasp binary success; we treat grasping as one phase of a
   multi-phase mission and quantify how continuous grasp *quality* — not
   binary grasp success — propagates unchanged through climb/transport
   (corr 0.9994) and dominates end-to-end delivery (5.1× delivery-failure
   ratio). (Do not claim "first"; phrase as "unlike this literature".)
4. **Grasp quality as task-success predictor** (Ferrari & Canny ICRA 1992;
   Dex-Net 2.0 RSS 2017; TOG-Net RSS 2018/IJRR 2020): task-oriented grasping
   selects among candidate grasps *before* contact using a quality model; in
   our system quality is realized stochastically by the contact event itself
   (R²=0.066 from pre-close state) and cannot be optimized at grasp time, so
   its value lies in post-hoc measurement from the policy's own observations
   as a class-specific gate.
5. **Modular failure diagnosis** (DAgger AISTATS 2011 compounding errors;
   REFLECT CoRL 2023; Gervet et al. Science Robotics 2023): prior work
   attributes end-to-end loss to per-module error rates or inter-module
   distribution shift; we localize the loss to a single continuous interface
   variable that passes every module's success check, and go beyond attribution
   to controlled falsification of the three standard no-retrain remedies.

## Five figures/tables

1. **Fig. 1 — System and mission pipeline.** (a) dual-purpose landing-gear
   gripper geometry (10 mm X vs 64 mm Y clearance around the 8 cm cube);
   (b) 8-phase mission state machine annotated with owning controller (RL
   flight 22D / analytical PID dock+climb / RL loaded 23D).
   *Source:* `data/final_presentation/v2_plots/gripper_geometry.png` +
   `pipeline_overview.png` (exist).
2. **Fig. 2 — The composition paradox.** (a) stage cascade: every stage passes
   at 81–100%, yet the per-stage product (0.986×0.905×0.809=72.2%) mis-models
   the loss because stages share a latent; (b) success vs inherited grasp depth
   at climb entry: 79.1% (n=220) vs 53.3% (n=135), same downstream components.
   *Source:* `figA_stage_cascade.png` + `figC_success_vs_depth.png`
   (regen: `scripts/make_composition_figs.py`).
3. **Fig. 3 — Characterizing the interface variable.** (a) irreversibility
   scatter (corr 0.9994, drift 0.055 mm); (b) 2×2 class-specificity (depth→
   delivery p=1e-12, tilt→climb p=7e-12, cross n.s.); (c) covariate shift
   (training bank 99.2% deep / mean −45.8 mm vs live 37% shallow / −25.8 mm).
   Optional inset: depth lock-in timing (figG) — depth is decided by contain
   250–280, before any gate can act.
   *Source:* `figE_irreversibility_scatter.png`, `figF_class_specificity_2x2.png`,
   `figB_covariate_shift.png`, `figG_depth_lockin.png` (exist).
4. **Fig. 4 + Table 1 — Three no-retrain intervention families: relocation,
   not recovery.** Existing 3-panel figure: (a) outcome composition per arm
   (69.0 / 38.0+52.1 gated / 63.0%); (b) first-attempt lock-depth
   distributions (ramp collapses the deep mode; regrasp draws from the
   unchanged first-attempt distribution); (c) risk relocation + conditional
   success (69.2 → 88.5 / 75.2%). Table 1 = full intervention ledger incl.
   earlier arms (action clip −19/−36 pp, barrier retrain −27/−2 pp, dock gain
   tune backfire 48→32%).
   *Source:* `figI_intervention_taxonomy.png` (exists;
   regen: `scripts/make_intervention_fig.py`) + `TABLES.md` T3 +
   `08_interventions_tested/intervention_comparison.csv`.
5. **Fig. 5 — Interface monitoring: class-specific failure prediction.** ROC
   curves for five signal→failure-class pairs (tilt→climb 0.777; depth→
   box-dropped 0.775 / too-tilted 0.764; vel_z→box-dropped 0.836; ang-vel→
   too-tilted 0.763) with operating point: tilt 16.6° captures 67% of climb
   failures at 5% false-reject.
   *Source:* `10_prescription_validation/fig2_gate_roc.png` + `stats.txt`
   (regen: `scripts/analyze_gate_validation.py`).

## Paper structure

- **Section 1 — Introduction**: composition-paradox hook; three questions
  (where is success lost / what kind of variable carries it / what can
  no-retrain interventions do); contributions C1–C5.
- **Section 2 — Related Work**: the five clusters above, each closed with its
  one-sentence difference.
- **Section 3 — Aerial Pick-and-Deliver System**: platform + gripper geometry,
  hybrid architecture (RL-dock failure history → analytical dock), training
  setup and grasp-state bank, 8-phase evaluation harness (incl. drain fix).
- **Section 4 — Diagnosing the Hidden Interface Variable**: metric blindness →
  depth split → irreversibility → orthogonal class-specificity → covariate
  shift; failure-label misattribution example (`too_tilted_delivery` is
  actually a shallow-depth failure: depth −16 mm, climb-entry tilt 7°).
- **Section 5 — No-Retrain Interventions at the Interface**: three families,
  same-seed evaluation, risk-relocation result; falsified analytic projection;
  the throughput-vs-certified-reliability frontier.
- **Section 6 — Discussion: Design Guidelines and Limitations**: C5 checklist;
  limitations (sim-only, single object/system, seed-1 arms with regrasp
  N=192, DR logging covers mass only, training-coverage confound for
  shallow-delivery failures).
- **Section 7 — Conclusion**.

## Evidence rules — claims that must NOT appear (judge-panel audit)

1. **No gate-only success rate exists.** Only `off/full/regrasp` JSONs are on
   disk; "gate-only 61%" (which two draft framings invented) is fabricated and
   arithmetically impossible alongside 52% rejection. The 88.5% / 52.1% / 1.8%
   numbers belong to the **shape+gate (full) arm** — always attribute them so,
   or run the gate-only arm (command below).
2. Say **"same-seed mission layouts (seed 1)"**, not "seed-paired" (regrasp ran
   N=192 vs 384; multi-seed pairing doesn't exist yet; `analyze_graspfix.py`
   paired stats need ≥2 seeds).
3. Regrasp verdict rests on **outcome-level evidence** (delivery mass
   18.0→10.4% p=0.020; dock mass 0.3→16.1%; e2e n.s. p=0.16). The depth panel
   shows *first-attempt* depths only (`lock_depths` is never re-logged after a
   retry) — do not claim "retries don't shift the depth distribution".
4. Full-arm depths are logged at the first contain≥285 crossing *mid-ramp*
   (ramp completes at 290), understating final seated depth — flag in the
   Fig. 4b caption or a reviewer will see a contradiction with 165 gate-passers.
5. DR null-check covers **mass only** (motor_kf/CoM/wind/plate-friction were
   not logged) — never claim the finding is robust to "mass, thrust, wind".
6. Scoped wording: **"no feedforward fix"**, not "fundamental ceiling";
   shallow-grasp delivery failure is **partly training-coverage** (99.2%-deep
   survivor bank), not pure physics.
7. Report **AUC, not Cohen's d** (d inflated by non-normality).
8. Pick ONE evaluation harness per table and state it: 72.2% (README
   500-mission harness) vs 69.0% (drain-fixed graspfix off, N=384) vs the
   ~66–68% unseeded band are different harnesses — reconcile explicitly once.
9. Decompose the full arm's 38.0%: 52.1 pp is gate rejections; the
   shaping-specific evidence is the depth collapse, not the 38% headline.
10. No "first to..." priority claims.

## Key verified statistics (2026-07-02, scipy on existing JSONs)

| Arm | N | End-to-end | 95% CI | Success \| accepted | Delivery-fail mass | Dock-fail(+rej) mass | Depth mean (1st att.) |
|---|---|---|---|---|---|---|---|
| off (baseline) | 384 | **69.0%** | [64.2, 73.4] | 69.2% | 18.0% | 0.3% | −22.3 mm (47.1% deep) |
| full (shape+gate) | 384 | **38.0%** | [33.3, 43.0] | 88.5% | 1.8% | 57.0% | −10.3 mm (0.0% deep) |
| regrasp (K=3) | 192 | **63.0%** | [56.0, 69.5] | 75.2% | 10.4% | 16.1% | −22.6 mm (49.2% deep) |

Fisher tests: e2e off-vs-full p<1e-4; off-vs-regrasp p=0.159 (n.s.);
delivery-mass off-vs-regrasp p=0.0199; conditional off-vs-full p<1e-4;
depth off-vs-full MW p=2.6e-44; off-vs-regrasp p=0.70 ("deep" = < −25 mm).

## Optional new-data experiments (USER-RUN ONLY — GPU required)

```bash
# 1) HIGHEST VALUE — gate-only arm (fills the one evidence hole in C3/C4;
#    lets Fig.4 attribute certification to the gate alone, ~30 min):
python scripts/eval_mission_headless.py --grasp-fix gate --seed 1 --target-missions 500 \
  2>&1 | tee logs/graspfix_run_gate_1.log

# 2) RECOMMENDED — multi-seed sweep (upgrades "same-seed" to true seed-paired
#    stats; analyze_graspfix.py paired Wilcoxon needs >=2 seeds; also widens
#    regrasp N beyond 192). Sequential, one Isaac instance:
SEEDS="1 2 3" MODES="off gate full regrasp" N=500 SKIP_PAIR=1 \
  bash scripts/run_graspfix_sweep.sh 2>&1 | tee logs/sweep_paper.log

# 3) OPTIONAL — attitude-gate ablation:
python scripts/eval_mission_headless.py --grasp-fix nogate --seed 1 --target-missions 500 \
  2>&1 | tee logs/graspfix_run_nogate_1.log
```

The paper is submittable WITHOUT these (evidence rules above keep every claim
inside existing data); #1 and #2 mainly harden C3 against reviewer round 2.

## Venue note

IJASC = *International Journal of Advanced Smart Convergence* (IIBC, Korea;
ISSN 2288-2847/2288-2855; KCI-indexed, not SCIE/Scopus; quarterly; English;
short applied papers ~6–10 pp two-column; ≥3 reviewers, ~1-month review).
Applied implementation-and-results register → lead with the system (C1), keep
theory light, keep length compact. (Distinct from IJASS, Springer/KSAS — if
the aerospace journal was intended, the same framing holds but Section 3
gains weight.)
