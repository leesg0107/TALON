# IJASC manuscript — assembly copy (revision 3)
#
# Formatting rules applied at text level (from IJASC_Format.doc):
#   * English throughout; sections numbered, initial capitals, left aligned
#   * Table captions ABOVE tables, "Table n. ..." ; figure captions BELOW figures, "Figure n. ..."
#   * Equations numbered consecutively with Arabic numerals; main symbols italic
#   * SI units, space between number and unit except °C and %
#   * Citations in brackets [1], [2,3]; reference list in citation order, IJASC format
#   * 4-6 keywords
# Symbols: s seating (mm) | h vertical clearance at the close command (mm)
#          beta jaw arrest angle (deg) | theta vehicle tilt (deg) | e_x strut-axis misalignment (mm)
# See docs/ijasc_format_notes.md for the full rule list.

---

## Title

**Exploring Grasp Depth in a Modular Aerial Pick and Deliver System**

## Authors / affiliations

(author block per template)

---

## Abstract

Delivery drones in service today are loaded and unloaded by people, so autonomy covers the
flight segment only. A vehicle that also picks up its own payload could run repeated transport
cycles unattended. I built such a system in simulation: a quadrotor whose two landing-gear jaws
close around the payload, driven by a modular pipeline of four phases — approach, dock, climb
and delivery — in which each phase hands the vehicle to the next through an explicit acceptance
test. Over 384 randomized missions the four phases lose 0, 1, 49 and 69 missions and 69.0%
complete. This paper asks where those losses come from. In a separately instrumented run of the
same configuration, handoffs that the dock acceptance test certifies identically differ in how
far the advancing jaw has driven the payload before being arrested by it — a quantity no
acceptance test measures. Sorted by it, the same downstream phases complete 79.1% against 50.9%
of missions, the delivery-phase rate separates into 91.1% and 53.9%, and poorly seated payloads
are dropped in flight 5.3 times as often. The quantity is fixed at the closure and is then
carried through the climb without change (r = 0.9994). Two conditions at the close command
account for it: a strut-axis alignment threshold that follows from the jaw geometry, below which
the engagement rate is 90.5% against 42.5% above it (p = 2 × 10⁻⁹), and an empirical vertical
clearance threshold that recovers the misaligned population. Three families of no-retrain
intervention, none of which alters either condition, move failures between phases without
raising completion. What transfers is the measurement procedure rather than the individual
numbers: it localizes a continuous interface quantity that per-phase success rates cannot see.

**Keywords:** Aerial manipulation, Modular pipeline, Grasp quality, Interface measurement,
Failure analysis

---

## 1. Introduction

Delivery drones in service today fly autonomously but are loaded and unloaded by people, so
every cycle still consumes human labour at both ends. If the vehicle could also pick up its
payload from a known place and set it down at another, the same aircraft could run repeated
transport cycles unattended and could collect items on the return leg.

Field systems that attempt this have been built and measured. At the MBZ International Robotics
Challenge, an aerial team detected 14 objects, contacted 13 of them, picked up 6 and delivered
4 — a servoing success rate of 93% but a gripping rate of 46% [1]. The winning team of the 2020
edition grasped 22 bricks and placed 13 [2]. Both are outdoor systems with onboard perception
and unmodelled targets, so their rates are not comparable with the simulated rates reported in
this paper and no such comparison is made here. What they establish is the shape of the loss:
the approach is nearly reliable, the grasp is where most missions are lost, and a further share
of payloads is lost after a grasp the system had counted as successful.

That pattern is the subject of this paper. I built the smallest system that can perform the
whole mission — a quadrotor whose two landing-gear jaws double as the gripper — and instrumented
it end to end in simulation. The system is organized the way such pipelines are normally built:
four phases run in sequence, two driven by learned policies and two by analytical controllers,
and each boundary carries an acceptance test that must be satisfied before control passes on.
Over 384 randomized missions, 69.0% complete, and the phase that records the most failures is
the last one.

That binary grasp success understates what a transport mission needs is not a new observation.
The GRASPA benchmark for arm grasping separates a binary acquisition score from a stability
score measured over a fixed post-grasp motion, precisely because the two diverge [3]. In the
field systems above, post-grasp losses were attributed to erroneous binary contact sensing [1],
and the 2020 winner added an abort that watches estimated mass and attitude during the grasp
interaction and gives up when the brick is held far from the centre of mass [2]. The phenomenon
is known and is worked around in practice. What is absent is a measurement of the quantity
itself, of how it reaches the phases that fail, and of what sets it. The contributions of this
paper are therefore:

* I measure a continuous interface quantity for every grasp in a complete aerial transport
  pipeline, show that it is fixed at the closure and carried through the following phase without
  change, and show that it separates mission completion by 28.2 percentage points among handoffs
  the pipeline certifies identically.
* I localize what accounts for it: two conditions at the close command, the first with a
  threshold that follows from the jaw geometry and is confirmed on 296 instrumented grasps
  (90.5% against 42.5% engagement, p = 2 × 10⁻⁹).
* I measure three families of no-retrain intervention at this interface under identical
  conditions and show that each redistributes failures between phases without raising mission
  completion, which is consistent with none of them altering either condition.

Section 2 places the work, Section 3 defines the system and its instrumentation, Section 4
states the hypotheses, Sections 5 and 6 report the measurements, and Section 7 discusses limits.

---

## 2. Related Work

**Aerial pick-and-transport systems.** Complete aerial pick-and-transport systems have been
built and reported with per-phase rates [1,2]; their numbers and the reason they are not
compared with this work are given in Section 1. One of them states that its nonlinear
model-predictive controller and visual-inertial estimator distinguished it from teams using PID
tracking control, and nonetheless attributes its post-grasp losses to contact sensing rather
than to tracking error [1]. Section 5.3 returns to this.

**Grippers built from vehicle structure.** Using structural members of the aircraft as the
gripper is an established design family: modularized landing gears have been used to grasp and
hook onto structures for perching and resting [4]. That work targets attachment to the
environment rather than payload transport and reports demonstrations rather than rates. Soft and
passively closing aerial grippers report per-object grasp rates instead, for example 9/10, 6/10
and 10/10 for three objects at 0.5 m/s with fully onboard perception, and observe that the kind
of contact achieved sets the retention capacity: an enveloping grasp sustained 2 kg while a
grasp relying on pinching force failed consistently at 250 g [5]. The present work measures a
distinction of that kind per attempt rather than per object.

**Grasp quality as a continuous quantity.** Benchmarks for arm grasping already separate
acquisition from retention: GRASPA scores a binary grasp success and, separately, the fraction
of a fixed post-grasp rotation trajectory completed without dropping the object [3]. Grasp
planning ranks candidate grasps by a continuous quality metric before contact [6]. Both presume
either a scoring stage after the grasp or a choice among candidates before it. The system
studied here has neither: the jaws are commanded shut once and are arrested wherever the payload
stops them, so quality is an outcome to be measured rather than an input to be selected, and the
question this paper adds is what happens to it downstream.

**Composition of phases.** Sequential composition certifies a chain of controllers by checking
that the outcome region of one lies inside the domain of the next [7], and the system studied
here implements exactly such predicates at its phase boundaries. Work on chaining learned skills
repairs composition at training time by shaping the terminal state distribution of one skill to
match the initial distribution of the next [8]; decomposing a transport task into sub-tasks with
their own sub-goals is a standard way to keep long-horizon transport tractable [9]. The mismatch
reported here belongs to this family but is not repaired downstream: the quantity is fixed by a
contact event, is carried through the next phase without change, and is absent from the training
distribution of the receiving policy.

**Deployment distribution shift.** A policy evaluated on states it did not see in training
degrades, and the error compounds along a trajectory [10]. Section 5.5 reports an instance.

---

## 3. The TALON System

### 3.1 Platform and jaw geometry

TALON is a simulated quadrotor whose landing gear also serves as its gripper, in the design
family of [4]. Two jaws are hinged to the vehicle body at *y* = ±20 mm and rotate about the
*x* axis, so they close along *y*; *x* is referred to below as the strut axis and *y* as the
closing axis. Each jaw consists of two members 110 mm long at *x* = ±50 mm, each 12 mm thick,
and a foot at their lower end pointing inward, 42 mm along the closing axis and 100 mm along the
strut axis. The payload is an 80 mm cube resting on a 300 mm × 300 mm pedestal, 500 mm tall. The
jaw joints are commanded from 50° from vertical, the landing pose, to −5°, the closed pose,
positive angles opening the jaw outward.

Two consequences of this geometry are used later:

1. In the commanded closed pose the two feet together cover 53 mm along the closing axis, less
   than the 80 mm payload width, so they cannot enclose it. The jaws are therefore arrested by
   the payload before reaching the commanded pose and the grasp is formed wherever they stop.
   This follows from jaws dimensioned for landing rather than for this payload, and it is what
   makes the stopping point a per-attempt quantity rather than a fixed property of the mechanism.
2. The inner faces of the members lie at ±44 mm from the vehicle centre and the payload
   half-width is 40 mm, so a strut-axis misalignment above 4 mm brings a member into
   interference with the payload.

### 3.2 Mission, controllers and completion

A mission has four phases. The approach phase flies the vehicle through waypoints to a position
above the payload under a policy trained with PPO [11]. The dock phase aligns and descends under
an analytical PID controller and commands the jaws shut. The climb phase lifts the loaded
vehicle to 1.0 m above the ground under a low-gain analytical controller, followed by a short
stabilization interval before the handoff to the delivery policy. The delivery phase flies the
loaded vehicle to a sequence of delivery waypoints under a second PPO policy trained with the
payload grasped. Control and physics run at 150 Hz.

A mission is **completed** when the vehicle reaches the final delivery waypoint within 0.30 m
while the payload is still held. It **fails** when the payload is lost — recorded when its
altitude falls below 0.30 m or it separates from the vehicle by more than 0.50 m — when the
vehicle tilt exceeds 70°, when its altitude falls below 0.10 m or its horizontal distance
exceeds 15 m, or when a phase exceeds its budget (12 s for the dock phase, 120 s for the
mission). Failing an acceptance predicate does not by itself end a mission: the vehicle stays in
the current phase and continues until that phase's budget is exhausted, at which point the
mission is recorded as a failure of that phase. The one exception is the gating arm of Section
6, in which a rejected handoff ends the mission and is recorded separately as a rejection.

The dock phase is analytical rather than learned because an earlier ablation of this system
trained a policy for the same descent-and-grasp on physically reactive objects and reached
24.8% success; the analytical controller was adopted on that basis. That figure is an
unpublished internal result of this project and is reported only as the design rationale.

### 3.3 Acceptance predicates and instrumentation

Each phase boundary is guarded by an acceptance predicate evaluated on the vehicle state. At the
dock boundary,

*A* = 1[*c* ≥ 200] ∧ 1[‖*p* − *p*_box‖ < 0.30 m] ∧ 1[*θ* < 15°] ∧ 1[‖*ω*‖ < 2 rad/s]  (1)

where *c* is a cumulative count of control steps in which more than half of the payload
projection lies inside the jaw envelope, *p* is the vehicle position, *θ* the tilt from vertical
and *ω* the body rate. The predicates at the other boundaries are given in Figure 1; the
approach and climb-exit predicates additionally admit a mission on a timeout fallback, so they
are permissive rather than strict.

Independently of the predicates, the interface is instrumented. Let *R* be the vehicle attitude
and *e*_y the unit vector along the closing axis. The **grasp depth**, or seating, of a grasp is

*s* = |*e*_yᵀ *R*ᵀ (*p*_box − *p*)|  (2)

the distance by which the payload centre lies off the vehicle centreline along the closing axis
at the moment the grasp is handed over. A large value is what a good grasp looks like in this
mechanism, for the following reason. The jaws cannot close symmetrically on an oversized payload
(Section 3.1): one jaw advances, drives the payload along the closing axis ahead of its foot,
and is arrested when the payload can move no further. The payload therefore ends up displaced
from the centreline by roughly the distance that jaw travelled, which is why *s* and the arrest
angle of the advancing jaw are two readings of one event (Section 5.2). Equation (2) is a
measurement only: *s* does not appear in (1) or in any other acceptance predicate. It is not
simply inherited from the approach — the same quantity measured at the close command accounts
for 32% of its variance, so about two thirds is produced during the closure.

### 3.4 Evaluation runs

All results come from one configuration of the Isaac Lab GPU-parallel robot learning framework,
the successor of the Orbit framework [12], evaluated in 128 parallel environments. Per-mission
randomization covers the vehicle mass scale (0.9–1.1), the motor thrust-coefficient scale
(0.85–1.15) and the phase of a slowly varying wind disturbance [13]; the payload mass, its
position on the pedestal, the contact friction and the centre-of-mass offset are held constant.
Every launched mission is run to completion, so no mission is truncated when a target count is
reached.

Three runs of this configuration are used and are never mixed within a figure or a table:

* **Baseline run**, 384 missions with a fixed seed, recording per-mission outcome labels. Used
  for the phase losses and as the reference arm for the interventions.
* **Handoff run**, 400 missions launched, 383 accepted at the dock boundary and 341 reaching the
  delivery phase, recording the full vehicle and payload state at the climb entry and at the
  delivery entry. No seed was set.
* **Closure run**, 300 missions launched and 296 producing a usable grasp, recording the vehicle
  state, the payload pose and both jaw angles at every step of the dock phase. No seed was set.

The two instrumented runs are unseeded and lose more missions inside the dock phase than the
baseline does — 17 of 400 against 1 of 384 — a difference this paper does not explain.
Downstream of the dock the runs agree closely: the handoff run survives the climb in 89.0% of
accepted handoffs against 87.2% in the baseline, and completes 70.2% of accepted handoffs
against 69.2%. All stratified results below use accepted handoffs only, so the dock-phase
difference does not enter them.

**[Figure 1 here]**

*Figure 1. Mission chain, the acceptance predicate at each phase boundary, and the seating s.
Numbers above the arrows are missions crossing that boundary and numbers below each phase are
missions lost in it, from the baseline run (N = 384). The approach and climb-exit predicates
also admit a mission on a timeout fallback. The seating is fixed when the jaws are arrested and
is then carried onward unchanged; it appears in none of the predicates.*

*Table 1. Per-phase losses of the baseline run (N = 384, fixed seed). A mission is attributed to
the phase in which its failure was recorded. Four of the 49 climb-phase losses occur in the
stabilization interval that follows the climb, during which the delivery policy is already
active.*

| Phase | Entered | Lost in this phase | Survived |
|---|---|---|---|
| Approach | 384 | 0 | 384 (100%) |
| Dock | 384 | 1 | 383 (99.7%) |
| Climb | 383 | 49 | 334 (87.2%) |
| Delivery | 334 | 69 | 265 (79.3%) |
| Mission completed | | | **265 / 384 = 69.0%** |

---

## 4. Hypotheses

Table 1 does not by itself locate the cause of the losses. A phase records a failure when a
mission ends in it, but the state that mission was in when it entered the phase was set earlier.
A per-phase rate is therefore a property of the inputs a phase receives as much as of the phase
itself, and per-module competence does not compose into system reliability [1,2,9]. I state
three hypotheses and test them in Sections 5 and 6.

**H1.** The mission losses are governed not by the competence of the downstream phases alone but
by the seating *s*, which is produced by the closure in the dock phase and inherited by the
phases that follow.

**H2.** The seating is accounted for by the geometric configuration at the instant the closure
is commanded:

*E* = 1[|*e*_x| < *e*\*] ∨ 1[*h* > *h*\*]  (3)

where *E* denotes an engaged grasp, *e*_x is the strut-axis misalignment and *h* is the height
of the jaw hinges above the payload top face, both at the close command. The geometry of Section
3.1 predicts *e*\* = 4 mm: below that misalignment no jaw member can interfere with the payload.
It does not predict *h*\*, and the mechanism by which vertical clearance restores engagement for
a misaligned approach is not established here; that term is empirical.

**H3.** The conditions in (3) are set before the payload arrests the jaws. An intervention that
does not alter *e*_x or *h* at the close command therefore cannot change the distribution of
*E*; it can only change what is done with a grasp whose quality is already fixed.

To test H3 I evaluate the three families of intervention available without retraining any
policy: shaping the closing motion, gating the handoff on measured quality, and re-grasping
after a poor grasp is detected. I do not evaluate a change to the terms of (3) themselves; that
experiment is identified in Section 7 as the next step.

---

## 5. Measurements

### 5.1 The losses are inherited, not generated where they are recorded

I take the 383 accepted handoffs of the handoff run, all of which satisfied (1), and sort them
by the seating *s* measured at the climb entry. Grasps with *s* > 25 mm are called engaged
(*n* = 220), those with *s* < 15 mm not engaged (*n* = 108), and the remainder mid (*n* = 55).
The cuts bracket the trough of the bimodal distribution in Figure 2(a); Section 5.2 shows that
the two modes correspond to a mechanical distinction.

**[Figure 2 here]**

*Figure 2. Handoff run. (a) Seating measured after acceptance: the dock acceptance predicate
admitted the full spread. (b) Outcome of the engaged and not-engaged groups, with Wilson 95%
intervals [14]; the mid group is omitted from (b) and reported in Table 2. The delivery-phase
rate is conditional on surviving the climb (n = 191 and 102).*

The two groups differ by 28.2 percentage points in mission completion, and the difference is not
produced in the climb phase: the not-engaged group survives the climb slightly more often than
the engaged group, 94.4% against 86.8%. The separation appears in the delivery phase, 91.1%
against 53.9%. The mid group sits between the two at 72.7%, so the relation is monotone across
the three groups.

Table 2 shows how the missions failed. The failure that dominates the not-engaged group is the
payload being lost in flight, at 5.3 times the rate of the engaged group.

*Table 2. Outcome of accepted handoffs by seating group (handoff run, 383 handoffs). Percentages
are of the group and sum to 100%: no mission of this run failed by timeout, by low altitude or
by excessive distance. A payload loss is counted in the phase in which it occurred, so the two
delivery columns exclude losses during the climb.*

| Group | *n* | Completed | Payload lost in delivery | Tilt > 70° in delivery | Lost in climb |
|---|---|---|---|---|---|
| Engaged (*s* > 25 mm) | 220 | 79.1% | 4.5% | 3.2% | 13.2% |
| Mid (15–25 mm) | 55 | 72.7% | 3.6% | 10.9% | 12.7% |
| Not engaged (*s* < 15 mm) | 108 | 50.9% | **24.1%** | 19.4% | 5.6% |

Three statements follow. First, the climb phase does not degrade with poor seating. Second, the
quantity that predicts the delivery outcome is fixed at the closure, two phases earlier. Third,
the aggregate delivery-phase rate is a mixture of two populations, 91.1% and 53.9%, and is
therefore not a property of the delivery policy alone.

The transfer through the climb is close to exact. Writing *s*_k for the seating at the climb
entry and *s*_{k+1} for the seating at the delivery entry over the 341 missions that reach
delivery,

*s*_{k+1} = *s*_k + *ε*,  *r* = 0.9994,  E|*ε*| = 0.055 mm  (4)

so the climb phase transports the seating without altering it. The delivery phase is not
instrumented at its exit, so the corresponding statement is made for one phase, not two. H1 is
supported.

### 5.2 What the inherited quantity is

Because the closed span of the feet is smaller than the payload (Section 3.1), the jaws cannot
reach the commanded closed pose; they are arrested by the payload. In the closure run the
commanded pose of −5° was reached in 0 of 296 grasps, and the arrest angle *β* of the advancing
jaw ranged from 6.0° to 42.2°.

**[Figure 3 here]**

*Figure 3. Closure run. Cross-sections in the closing plane, drawn from the CAD geometry at the
group-median configuration; the body tilt at that instant (mean 11.6°, median 9.0°) is omitted
for clarity. The jaw members lie outboard of the section plane at x = ±50 mm and are therefore
not drawn; they interfere with the payload when |e_x| exceeds 4 mm, which is the mechanism of
H2. In every grasp of this run the same jaw advanced further than the other, so (b) and (c) show
one jaw at its arrest angle and the other near its open pose. (a) Closure commanded with both
jaws at 50°. (b) Engaged outcome. (c) Not-engaged outcome. (d) Seating against arrest angle.*

The arrest angle and the seating are two readings of the same event: their Spearman correlation
is −0.91 over the 296 grasps. Reconstructing the foot-to-payload distance from the part geometry
separates the two groups. Taking a contact as a reconstructed gap below 2 mm, which absorbs the
error of the reconstruction, 97.9% of engaged grasps have at least one foot in contact with the
payload while 97.9% of not-engaged grasps have none; the median gaps are −3 mm and +17 mm. Under
the stricter criterion of a gap below 0 mm the engaged figure falls to 93.2% and the not-engaged
figure is unchanged. The median arrest angles are 7.5° and 32.5°.

The seating is therefore a continuous measure of a mechanical event: whether the advancing jaw
carried its foot as far as the payload before being arrested. Because the closure is one-sided,
an engaged grasp is not an enveloping one; it is a payload held between one foot and the
opposing jaw. What distinguishes it from a not-engaged grasp is that a foot has reached the
payload at all, rather than the payload being retained by the jaw faces alone. Contacts of the
latter kind are known to sustain far smaller loads: in a comparable aerial gripper an enveloping
grasp sustained 2 kg while a grasp relying on pinching force failed consistently at 250 g [5].
This is consistent with the failure modes of Table 2, in which the not-engaged group is lost
predominantly by losing the payload in flight rather than by any failure of the climb.

### 5.3 What accounts for the event

Taken on its own, the vertical clearance *h* separates the outcomes only weakly and its marginal
relation to engagement is not monotone. The non-monotonicity is a confound: at the close command
the 34 grasps with the smallest clearance (*h* < 38 mm, the lowest 11.5%) are also the best
aligned, with a median |*e*_x| of 4.5 mm against 13.4 mm for the rest. Stratifying by both
variables resolves it.

**[Figure 4 here]**

*Figure 4. Closure run, n = 296. (a) Engagement rate over terciles of strut-axis misalignment
and vertical clearance. (b) The same grasps at the close command. The dashed line is the
interference threshold predicted by the jaw geometry, |e_x| = 4 mm.*

The alignment threshold predicted in Section 3.1 is confirmed directly: grasps with
|*e*_x| < 4 mm engage in 90.5% of cases (95% interval 78–96, *n* = 42) against 42.5% for the
rest (37–49, *n* = 254), Fisher exact *p* = 2 × 10⁻⁹. The structure of (3) is visible in the
stratification. When the misalignment is below 7 mm the grasp engages in 68–82% of cases
irrespective of clearance; when it exceeds 17 mm the grasp engages only when the clearance
exceeds 49 mm, where the rate rises from 0% to 73.5%. A logistic classifier on the two variables
reaches an out-of-fold area under the ROC curve of 0.848 under stratified 5-fold
cross-validation [15], against 0.637 and 0.699 for the variables individually. The clearance
threshold of 49 mm is the upper tercile boundary and is therefore empirical and
binning-dependent, unlike *e*\*.

This raises the obvious engineering objection. The mean strut-axis misalignment of the dock
controller, 13.4 mm, is more than three times the 4 mm tolerance implied by the jaw geometry, so
one could conclude that the controller simply needs to be more accurate. Held to |*e*_x| < 4 mm
it would indeed engage 90.5% of the time, and that is a legitimate design direction for this
platform. Two observations qualify it. Within this system, alignment is not the only route:
(3) is a disjunction, and the clearance term recovers the misaligned population without any
change to the tracking loop. Across systems, the aerial team whose tracking architecture was the
most elaborate of its competition — visual-inertial estimation with nonlinear model-predictive
control, contrasted by its own authors with the PID tracking used by other teams — still
reported a 46% gripping rate and attributed its post-grasp losses to contact sensing rather than
to tracking error [1]. Tracking accuracy is one lever among several, and on the evidence
available it is not by itself decisive for this class of loss.

### 5.4 Which quantity governs which phase

Seating is not the only quantity that crosses the dock boundary. Table 3 separates its effect
from that of the vehicle tilt at the climb entry, using the 15° threshold of predicate (1).

*Table 3. Per-phase attribution of two interface quantities (handoff run). p-values are Fisher
exact tests; no correction for multiple comparisons is applied.*

| Test | Rates | *p* |
|---|---|---|
| Tilt → loss in climb, within engaged | 5.3% → 57.6% | 7 × 10⁻¹² |
| Seating → loss in climb, within low tilt | 5.3% vs 3.1% | 0.55 |
| Seating → loss in delivery, within low tilt | 7.3% → 50.0% | 3 × 10⁻¹⁵ |
| Tilt → loss in delivery, within engaged | 7.3% → 28.6% | 0.025 |

**[Figure 5 here]**

*Figure 5. Handoff run, engaged and not-engaged groups only. Failure rate by seating group and
tilt group, with Wilson 95% intervals [14]. Cells in the upper row contain 8 to 33 handoffs and
are correspondingly uncertain; counts in (b) are smaller than in (a) because (b) is conditional
on surviving the climb.*

The climb phase fails on tilt and is indifferent to seating. The delivery phase fails on
seating: within the low-tilt stratum the loss rate rises from 7.3% (95% interval 4–12) to 50.0%
(40–60). The additional effect of tilt on delivery rests on 14 handoffs, would not survive a
correction for the four comparisons in the table, and runs the other way in the not-engaged
column over 8 handoffs; it should be read as unresolved.

This also explains the ordering noted in Section 5.1: within the low-tilt group the two seating
groups survive the climb equally, and the engaged group survives less often overall because
high-tilt handoffs are more frequent among engaged grasps, 15% against 10%.

### 5.5 The receiving policy was trained on a filtered distribution

The delivery policy was trained from a bank of 4996 grasp states collected under this same
configuration by keeping only grasps that survived a climb. That bank is 99.2% engaged with a
mean seating of 45.8 mm, whereas the live pipeline hands the policy a mean of 25.8 mm with 29.9%
of arrivals not engaged. A policy evaluated on states absent from its training distribution
degrades, and collecting more data of the same biased kind does not repair it [10]. The delivery
failures of the not-engaged group therefore mix a physical component with a training-coverage
component, and the present data cannot separate them.

---

## 6. Interventions at the Interface

I evaluated the three no-retrain families of Section 4 with the same seed as the baseline arm.
Shaped closing ramps the jaw command linearly over the arrest window, from the open pose at
*c* = 225 to the closed pose at *c* = 290, instead of commanding the closed pose in one step.
Handoff gating adds the term 1[*s* > 25 mm] to predicate (1) and rejects handoffs that fail it.
Re-grasping applies the same test after the closure and, on failure, reopens the jaws and
re-docks, up to three attempts. Shaping and gating were evaluated together in one arm, so their
individual contributions are not separated.

*Table 4. Outcome of the intervention arms. Shares are percentages of all missions of that arm;
conditional completion is over missions whose handoff was accepted, which for the arms without a
gate means missions that did not fail inside the dock phase.*

| Arm | *N* | Completed | Rejected | Lost in dock | Lost in climb | Lost in delivery | Completed \| accepted |
|---|---|---|---|---|---|---|---|
| Baseline | 384 | 69.0% | — | 0.3% | 12.8% | 18.0% | 69.2% |
| Shaped close + gate | 384 | 38.0% | 52.1% | 4.9% | 3.1% | 1.8% | 88.5% |
| Re-grasp (K = 3) | 192 | 63.0% | — | 16.1% | 10.4% | 10.4% | 75.2% |

**[Figure 6 here]**

*Figure 6. (a) Outcome composition of each arm. (b) Share of all missions of the arm lost in the
dock and delivery phases; rejections appear in (a) and are excluded here, so the gate arm's
delivery share is depressed by the missions it never admitted. (c) Throughput against
conditional reliability, with Wilson 95% intervals on the conditional rate [14]; the three arms
are distinct interventions, not samples of a continuum.*

Each intervention did what it was designed to do at the delivery phase. The delivery-phase loss
fell from 18.0% of missions to 1.8% under gating and to 10.4% under re-grasping, the latter
significant against the baseline (*p* = 0.020). Neither raised mission completion. Under gating
the reduction is in part a denominator effect, since 52.1% of missions never entered the
delivery phase; the conditional rate is the honest comparison, and it rose from 69.2% to 88.5%.
The losses that left the delivery phase reappeared upstream: the dock-phase loss rose from 0.3%
to 4.9% and 16.1% respectively, and of the 52.1% refused, 46.4 percentage points were refused on
the seating term and 5.7 on the pre-existing tilt term.

Re-grasping is the clearest case. It reduced the delivery loss significantly yet completed 63.0%
against 69.0%, a difference that is not significant (*p* = 0.16), because reopening the jaws
lost the payload during the dock in 4.7% of missions and exhausted the retry budget or the phase
timeout in a further 11.5%.

What the arms buy is certainty about the missions they accept. Defining

*T* = P(completed),  *R* = P(completed | handoff accepted)  (5)

the arms lie at (*T*, *R*) = (69.0, 69.2), (63.0, 75.2) and (38.0, 88.5). Conditional
reliability rises as throughput falls, and no arm improves on both.

This is the behaviour H3 predicts. None of the three alters *e*_x or *h* at the close command:
gating and re-grasping act after the jaws have been arrested, and shaped closing changes the
command profile during the arrest window without changing the geometry that (3) depends on. Its
effect on the seating distribution is nonetheless the largest of the three — the engaged share
of first attempts falls from 47.1% to 0.0% — although the two arms log the seating at different
points of the closure, so the magnitude of that shift is not directly comparable. The field
practice of Section 1 is consistent with this reading from the other direction: the winning
MBZIRC 2020 system aborts a grasp *during* the interaction, before committing to lift, when its
estimated mass and attitude indicate that the brick is badly held [2] — an intervention placed
before the quantity is fixed rather than after.

---

## 7. Discussion

**What the measurements support.** The losses of this system are governed by a quantity produced
by a contact event, carried through the following phase without change, and absent from every
acceptance predicate in the pipeline. Because that quantity is not measured, the per-phase rates
cannot be read as component properties: the delivery phase reports 79% while operating at 91.1%
on the grasps it can hold and 53.9% on those it cannot. Part of that gap is a property of the
delivery policy, which was never trained on the second population (Section 5.5); the point is
that neither part is visible in the per-phase rate.

**Design consequences.** Instrument continuous interface quality alongside the boolean
acceptance test, and test a candidate quantity for irreversibility before investing in the
downstream phases, since a quantity with the propagation of (4) cannot be repaired later. Treat
interface quality as a training-distribution variable, and place a quality check before the
commitment rather than after it, as (3) and the field practice of [2] both indicate.

**Measurability on hardware.** Equation (2) requires the payload pose, which a real vehicle does
not have. The arrest angle *β* does not: it is a joint encoder reading, it is available on any
actuated gripper, and it correlates with the seating at ρ = −0.91 (Section 5.2). An onboard
implementation of the measurement proposed here would use *β* rather than (2).

**Limitations.** The results are from simulation, one gripper and one payload. The 4 mm and
49 mm thresholds are properties of that pair; a compliant gripper or a closed geometry matched
to the payload would widen them. The contact model is the simulator default and the contact
friction is held constant, so the retention argument of Section 5.2 rests on the failure modes
of Table 2 and on [5] rather than on a friction sweep. The payload position on the pedestal is
not randomized, so the misalignment distribution of Figure 4(b) is produced by the approach and
by wind and mass dispersion only, and would be wider under realistic placement uncertainty. The
intervention arms use a single seed and the re-grasp arm has half the sample of the others; the
128 environments of a batch share a wind field, so the intervals and the p-values reported are
both optimistic. The two instrumented runs lose more missions inside the dock phase than the
baseline (Section 3.4), which this paper does not explain. In every grasp of the closure run the
same jaw advanced further than the other, which is a systematic asymmetry of the model rather
than a stochastic one; the seating is informative because of that asymmetry, so a
solver-independent replication is needed before either the asymmetry or the magnitude of *s* is
treated as a property of the physical mechanism rather than of this model. Finally, the
mechanism that retains those not-engaged grasps which nonetheless complete their mission is not
established.

**Next steps.** Two experiments follow from (3). The first moves the clearance term by raising
the hold setpoint of the dock controller and measuring the engagement rate of the misaligned
population; the stratification of Figure 4 suggests a large effect, but raising the setpoint
also changes the descent dynamics, so the observational strata are a hypothesis rather than a
prediction. The second rebuilds the delivery training bank from the handoff distribution the
pipeline actually produces and fine-tunes the delivery policy, which would separate the physical
and training-coverage components of Section 5.5.

---

## 8. Conclusion

I measured a complete aerial pick-and-deliver pipeline end to end and located the cause of its
losses. They are not generated by the phase in which they are recorded. They are carried by the
seating of the grasp, a continuous quantity fixed when the jaws are arrested by the payload,
carried through the climb without change, and measured by no acceptance test in the pipeline.
Sorted by it, handoffs the pipeline certifies identically complete 79.1% against 50.9% of
missions, and poorly seated payloads are lost in flight 5.3 times as often. Two conditions at
the close command account for the quantity, the first with a threshold that follows from the jaw
geometry and separates engagement 90.5% against 42.5%, and three families of no-retrain
intervention — none of which alters either condition — redistribute failures between phases
without raising completion. What transfers is the measurement procedure: a continuous interface
quantity, tested for irreversibility, stratified against outcome, and traced to the conditions
that set it.

---

## Acknowledgement

(per template, not numbered)

---

## References

[1] R. Bähnemann, M. Pantic, M. Popović, D. Schindler, M. Tranzatto, M. Kamel, M. Grimm,
J. Widauer, R. Siegwart, and J. Nieto, "The ETH-MAV Team in the MBZ International Robotics
Challenge," Journal of Field Robotics, Vol. 36, No. 1, pp. 78-103, January 2019.
DOI: 10.1002/rob.21824.

[2] T. Baca, R. Penicka, P. Stepan, M. Petrlik, V. Spurny, D. Hert, and M. Saska, "Autonomous
Cooperative Wall Building by a Team of Unmanned Aerial Vehicles in the MBZIRC 2020 Competition,"
Robotics and Autonomous Systems, Vol. 167, 104482, September 2023.
DOI: 10.1016/j.robot.2023.104482.

[3] F. Bottarel, G. Vezzani, U. Pattacini, and L. Natale, "GRASPA 1.0: GRASPA is a Robot Arm
graSping Performance BenchmArk," IEEE Robotics and Automation Letters, Vol. 5, No. 2,
pp. 836-843, April 2020. DOI: 10.1109/LRA.2020.2965865.

[4] K. Hang, X. Lyu, H. Song, J. A. Stork, A. M. Dollar, D. Kragic, and F. Zhang, "Perching and
Resting - A Paradigm for UAV Maneuvering with Modularized Landing Gears," Science Robotics,
Vol. 4, No. 28, eaau6637, March 2019. DOI: 10.1126/scirobotics.aau6637.

[5] S. Ubellacker, A. Ray, J. M. Bern, J. Strader, and L. Carlone, "High-Speed Aerial Grasping
Using a Soft Drone with Onboard Perception," npj Robotics, Vol. 2, No. 1, 5, August 2024.
DOI: 10.1038/s44182-024-00012-1.

[6] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. Aparicio Ojea, and
K. Goldberg, "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and
Analytic Grasp Metrics," in Proc. Robotics: Science and Systems (RSS), July 2017.
DOI: 10.15607/RSS.2017.XIII.058.

[7] R. R. Burridge, A. A. Rizzi, and D. E. Koditschek, "Sequential Composition of Dynamically
Dexterous Robot Behaviors," The International Journal of Robotics Research, Vol. 18, No. 6,
pp. 534-555, June 1999. DOI: 10.1177/02783649922066385.

[8] Y. Lee, J. J. Lim, A. Anandkumar, and Y. Zhu, "Adversarial Skill Chaining for Long-Horizon
Robot Manipulation via Terminal State Regularization," in Proc. 5th Conference on Robot
Learning (CoRL), PMLR Vol. 164, pp. 406-416, November 2021.

[9] G. Eoh, "Deep-Reinforcement-Learning-Based Object Transportation Using Task Space
Decomposition," Sensors, Vol. 23, No. 10, 4807, May 2023. DOI: 10.3390/s23104807.

[10] S. Ross, G. Gordon, and D. Bagnell, "A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning," in Proc. 14th International Conference on Artificial
Intelligence and Statistics (AISTATS), PMLR Vol. 15, pp. 627-635, April 2011.

[11] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy
Optimization Algorithms," arXiv:1707.06347, July 2017.

[12] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo,
H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg, "Orbit: A Unified
Simulation Framework for Interactive Robot Learning Environments," IEEE Robotics and Automation
Letters, Vol. 8, No. 6, pp. 3740-3747, June 2023. DOI: 10.1109/LRA.2023.3270034.

[13] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain Randomization
for Transferring Deep Neural Networks from Simulation to the Real World," in Proc. IEEE/RSJ
International Conference on Intelligent Robots and Systems (IROS), pp. 23-30, September 2017.
DOI: 10.1109/IROS.2017.8202133.

[14] E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference,"
Journal of the American Statistical Association, Vol. 22, No. 158, pp. 209-212, June 1927.
DOI: 10.1080/01621459.1927.10502953.

[15] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel,
P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher,
M. Perrot, and E. Duchesnay, "Scikit-learn: Machine Learning in Python," Journal of Machine
Learning Research, Vol. 12, pp. 2825-2830, 2011.
