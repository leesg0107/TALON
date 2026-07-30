# IJASC manuscript — assembly copy (revision 6)
#
# Formatting rules applied at text level (from IJASC_Format.doc):
#   * English throughout; sections numbered, initial capitals, left aligned
#   * Table captions ABOVE tables, "Table n. ..." ; figure captions BELOW figures, "Figure n. ..."
#   * Equations numbered consecutively with Arabic numerals; main symbols italic
#   * SI units, space between number and unit except °C and %
#   * Citations in brackets [1], [2,3]; reference list in citation order, IJASC format
#   * 4-6 keywords
# Symbols: d payload centre in the vehicle body frame (mm) | s grasp depth at the handoff (mm)
#          s_0 initial offset at the close command (mm) | e_x strut-axis misalignment (mm)
#          h vertical clearance at the close command (mm) | beta jaw arrest angle (deg)
#          theta vehicle tilt (deg)
# Figure number -> file in data/paper_figures/ (numbers and filenames DO NOT match; use this map):
#   Figure 1 = fig1_mechanism      Figure 2 = fig1_pipeline_predicates
#   Figure 3 = fig3_stall_and_engagement   Figure 4 = fig2_outcome_divergence
#   Figure 5 = fig4_engagement_conditions  Figure 6 = fig6_interventions
# Tables: 1 Failure conditions | 2 Evaluation runs | 3 Outcome by group |
#         4 Climb-loss 2x2 (depth x tilt) | 5 Intervention arms
# See docs/ijasc_format_notes.md for the full rule list.

---

## Title

**Exploring Grasp Depth in a Modular Aerial Pick and Deliver System**

## Authors / affiliations

(author block per template)

---

## Abstract

A drone that picks up its own payload could run transport cycles without a person at either end.
I built one in simulation: a quadrotor whose two landing-gear jaws serve as its gripper, run as
a pipeline of four phases (approach, dock, climb, delivery), each handing the vehicle on through
an explicit acceptance test. Across 384 randomized missions it completes 69.0%, the last phase
records the most failures, and this paper traces the largest share of them to a quantity set two
phases earlier. The jaws are landing gear, and closed they span less than the payload, so they
cannot shut around it. Closing, they first centre the payload between them; then one jaw keeps
going, pushing it sideways until it can push no further. That push distance, which equals how
much further the jaw closed after contact, is what the paper calls the grasp depth. Driven far
enough, the jaw's foot passes underneath the payload, which then rests on it; short of that, the
payload is held between the jaw faces alone. Nothing in the pipeline distinguishes the two. In
instrumented reruns of the same configuration, handoffs that passed one and the same acceptance
test complete their missions at 79.1% when the grasp was deep against 50.9% when it was shallow
(19.7 points under a plain median split), and the shallow ones drop their payload in flight 5.3
times as often. The depth is settled the moment the jaws stop, and the climb, the one phase
measured at both of its ends, carries it forward essentially unchanged (mean absolute change 0.055 mm;
one mission in 341 changes group). Three quantities readable at the close command predict
whether a grasp will end deep enough to rest on a foot: the vehicle's misalignment across the
closing direction, the payload's offset along it, and the height above the payload. The first
carries a tolerance the jaw geometry fixes in advance: 90.5% of grasps inside it reach the foot,
against 42.5% outside (p = 2 × 10⁻⁹). Two intervention arms that between them reshape, gate, and
retry the grasp never raise completion; failures only move to earlier phases. What transfers is the
measurement procedure, not the numbers.

**Keywords:** Aerial manipulation, Modular pipeline, Grasp quality, Interface measurement,
Failure analysis

---

## 1. Introduction

Delivery drones in service today fly autonomously but are loaded and unloaded by people, so
every cycle still consumes human labour at both ends. A vehicle that could pick up its own
payload and set it down again would run those cycles unattended; this paper studies the
pick-and-carry portion, leaving release itself out of scope (Section 7). That means adding a
grasp to the flight, and the grasp is where such systems lose their missions. At the MBZ
International Robotics Challenge (MBZIRC), an aerial team detected 14 objects, contacted 13 of
them, picked up 6 and delivered 4 — a servoing success rate of 93% but a gripping rate of 46%
[1]. The team that won the wall-building challenge of the 2020 edition grasped 22 bricks across
its trials and placed 13 [2]. Both are outdoor systems with onboard perception and unmodelled
targets, so their rates are not comparable with the simulated rates reported in this paper.
Between them they locate the losses: a few on the approach, most at the grasp itself, and more
after the grasp was recorded as good — dropped in flight in one system [1], lost at placement in
the other [2].

That pattern is the subject of this paper. I built a minimal system that performs the whole
mission — a quadrotor whose two landing-gear jaws double as the gripper — and instrumented it
end to end in simulation. The system is organized the way such pipelines are normally built:
four phases run in sequence, two driven by learned policies and two by analytical controllers,
and each boundary carries an acceptance test, evaluated before control passes on. Over 384
randomized missions, 69.0% complete (95% interval 64.2–73.4%), and the phase that records the
most failures is the last one.

A breakdown of that kind invites the reading that the last phase is the weakest one. That
reading is available only because each phase is scored on its own outcomes. A phase records a
failure when a mission ends inside it, but the state that mission was in when it entered was set
earlier; a per-phase rate is therefore a property of the inputs a phase receives as much as of
the phase itself. Competence measured module by module does not add up to reliability measured
end to end. To get behind such a breakdown one has to name a quantity that crosses a boundary,
measure it on every crossing, and follow it forward.

In this system such a quantity exists, and it exists because the gripper is landing gear that
was asked to do a second job. Two legs that fold up under an aircraft are not two jaws that meet
around a box. They shut until the box stops them, and where the box comes to rest between them —
pushed off centre by whichever jaw closed further — is not anything the pipeline commanded. No
controller chooses that resting position and no acceptance test inspects it, yet every phase
after the dock flies with it. This paper measures it as the grasp depth — and why one jaw, given
a mirrored command, always closes further than the other is itself part of what Section 5.1
measures.

It is not new to observe that a binary grasp result understates what a transport mission needs;
Section 2 collects the benchmarks and field systems that already work around it. What is absent
is a measurement of the quantity itself: how large it is, how it reaches the phases that fail,
and what sets it. The contributions of this paper are therefore:

* I measure a continuous interface quantity for every grasp in a complete aerial transport
  pipeline and show that the closure fixes it and the next phase leaves it untouched. Among
  handoffs the acceptance test scores identically, the two populations it separates are 28.2
  percentage points apart in mission completion.
* I identify what accounts for it: three conditions at the close command, the first with a
  threshold derived from the part dimensions before the data were examined and then confirmed on
  296 instrumented grasps (90.5% against 42.5% engagement, p = 2 × 10⁻⁹).
* I measure the three interventions available at this interface without retraining. None raised
  mission completion — the gate could not have by construction — and what they trade is
  throughput for conditional reliability: failures leave the delivery phase and reappear
  earlier.

Those three contributions follow one procedure, and the procedure is what transfers; the
Conclusion states its four steps.

Section 2 places the work, Section 3 defines the system and its instrumentation, Section 4
states the hypotheses, Sections 5 and 6 report the measurements, Section 7 discusses what they
support and what bounds them, and Section 8 concludes.

---

## 2. Related Work

**Aerial pick-and-transport systems.** The two field systems quoted in Section 1 report their
results phase by phase, and one detail in the first is worth pausing on. That team credits its
visual-inertial estimation and nonlinear model-predictive control with setting it apart from
rivals mostly running geometric or PID tracking — yet of its two payloads lost after a certified
grasp, one was lost to a false contact reading: the grasp was registered as successful and was
not [1]. A pipeline can lose payloads it has already certified, and when it does, what failed is
the certificate.

**Grippers and grasp quality.** Using the aircraft's structure as the gripper is an established
design family — modularized landing gears have grasped and hooked structures for perching and
resting [3], though for attachment rather than transport, and with demonstrations rather than
rates. Soft aerial grippers report per-object rates (9/10, 6/10, 10/10 at 0.5 m/s with onboard
perception), and their authors observe that the kind of contact achieved, not the fact of it,
sets how much load the grasp holds [4]; this paper measures such a distinction per attempt.
Benchmarks and planners already treat grasp quality as more than a bit — GRASPA scores
acquisition and retention separately [5], and grasp planning ranks candidates by a continuous
metric before contact [6] — but both presume a scoring stage or a menu of candidates. The system
here has neither: the jaws are commanded shut once and are stopped wherever the payload stops
them, so quality is an outcome to be measured, and the question this paper adds is what happens
to it downstream.

**Composition of phases.** Sequential composition certifies a chain of controllers by checking
that the outcome region of one lies inside the domain of the next [7], and the system studied
here implements exactly such predicates at its phase boundaries. Work on chaining learned skills
repairs composition at training time by shaping the terminal state distribution of one skill to
match the initial distribution of the next [8]; decomposing a transport task into sub-tasks with
their own sub-goals has been used to keep a long transport task tractable [9]. The mismatch
reported here belongs to this family but is not repaired downstream: the quantity is fixed by a
contact event, is carried through the next phase without change, and is absent from the training
distribution of the receiving policy. A policy evaluated on states it did not see in training
degrades, and the error compounds along a trajectory [10]. Section 5.2 reports an instance.

---

## 3. The TALON System

### 3.1 Platform and jaw geometry

TALON is a simulated quadrotor whose landing gear also serves as its gripper, in the design
family of [3]. Two jaws are hinged to the vehicle body at *y* = ±20 mm and rotate about the *x*
axis at *z* = −22.5 mm below the body reference point, so they close along *y*; *x* is referred
to below as the strut axis and *y* as the closing axis. Each jaw consists of two members, or
struts, 110 mm long and 12 mm thick, at *x* = ±50 mm, joined at their lower ends by an
inward-pointing foot, 42 mm across the closing axis with its centre 15 mm inboard of the member
line, and 100 mm along the strut axis. The vehicle has a mass of 1.12 kg and the payload is an
80 mm cube of 0.20 kg, 18% of the vehicle mass, resting on a 300 mm × 300 mm pedestal, 500 mm
tall.

Jaw angles are measured outward from vertical, so a larger angle means a more open jaw. The
joints rest at 45° for landing. Through the approach and the descent they are held at 50°, the
open pose, and at the close command they are driven to −5°, the closed pose. Both joints receive
the same commanded angle at every control step and their hinge axes are mirrored, so the
commanded motion is a mirror image on the two sides.

**[Figure 1 here]**

*Figure 1. The vehicle and its jaws, drawn from the URDF model. (a) The open pose, jaws at
50° from vertical. (b) The same vehicle descended over the payload, jaws still open. (c) Section
in the closing plane, showing the commanded closed pose (dashed, not reachable) and a typical
pose where the jaws actually stop. The vertical arrow marks the clearance *h*, the height of
the jaw hinges above the payload's top face, read at the close command (Section 3.3). At the commanded −5° each foot's inner edge would reach 26.6
mm past the vehicle centreline (hinge at ±20 mm, foot centred 15 mm inboard of the member,
half-width 21 mm), so the two feet together would cover only 53 mm of the closing axis — less
than the 80 mm payload — and the jaws cannot close around it. (d) Section along the strut axis.
The hatched bands are the strut-axis extent swept by a jaw member over its whole travel; because
the hinge axis is x, they do not move as the jaw rotates. Their inner faces lie at ±44 mm from
the vehicle centre against a payload half-width of 40 mm, so at a misalignment below 4 mm (left)
no member can reach the payload at any jaw angle, while at the measured mean misalignment of
13.4 mm (right) the bands overlap the payload. Overlap is necessary for interference, not
sufficient.*

Everything below rests on Figure 1(c): the jaws cannot enclose the payload, so the grasp is
whatever configuration they were stopped in. That is a consequence of dimensioning the jaws for landing, and it is why the grasp has to
be described by a quantity that is measured rather than by one that was designed.

### 3.2 Mission, controllers and completion

A mission has four phases. The approach phase flies the vehicle through waypoints to a position
above the payload under a policy trained by reinforcement learning, using proximal policy
optimization (PPO) [11]. The dock phase aligns and descends under an analytical PID controller
and commands the jaws shut. The climb phase lifts the loaded vehicle to 1.0 m above the ground
under a low-gain analytical controller, then holds it steady for up to 1.0 s before control
passes to the delivery policy. The delivery phase flies the loaded vehicle to a sequence of
delivery waypoints under a second PPO policy, trained with the payload already grasped. Control
and physics run at 150 Hz.

Three boundaries separate the four phases, and only one is studied here: the moment the dock
phase hands the loaded vehicle to the climb phase. At that moment the grasp exists and the
vehicle is released to fly. This paper calls that moment *the interface*, and one vehicle's
passage through it *a handoff*. Both words are used in that sense and no other; the
climb-to-delivery transition is called the delivery policy switch.

A mission is *completed* when the vehicle comes within 0.30 m of the final delivery waypoint
with the payload still held; the delivery target is drawn uniformly within ±3 m of the arena
centre and approached through three waypoints that climb to 2.5 m. It *fails* under the
conditions of Table 1 — and those conditions are not uniform across phases. The tilt, altitude
and range tests run only while a learned policy flies the vehicle, so the dock and the climb can
fail only by losing the payload, by the climb's own crash test, or by timeout. Per-phase loss
counts therefore compare phases with different failure criteria as well as different inputs — a
second reason, alongside inherited state, why such counts are not component properties.

*Table 1. Failure conditions and the phases in which each is evaluated. A phase budget ends the
mission in that phase.*

| Condition | Threshold | Evaluated during |
|---|---|---|
| Payload lost (low) | payload below 0.30 m above the ground | all phases |
| Payload lost (separated) | payload more than 0.50 m from the vehicle | all phases |
| Vehicle tilt | above 70° | approach, delivery |
| Vehicle altitude | below 0.10 m | approach, delivery |
| Vehicle range | more than 15 m from the launch point | approach, delivery |
| Climb crash test | vehicle below 0.30 m, 0.4 s after entering the climb | climb |
| Phase budget | 12 s for the dock; 120 s for the whole mission | dock; mission |

The climb's crash test accounts for 43 of the 49
climb-phase losses of the baseline run (four more occur in the stabilization interval that
follows the climb and two are timeouts). It fires when the loaded vehicle is found below 0.30 m
— beneath the top of the 0.50 m pedestal — 0.4 s or more after entering the climb: the signature
of a vehicle that left the pedestal with its payload and could not hold altitude. It is a
fallen-vehicle detector, not a rate-of-climb requirement. One further behaviour is shared by
every arm, including the baseline: a closure that rises without the payload — the vehicle
climbing alone while the payload stays on the pedestal — is detected within 0.2 s of entering
the climb and returns the vehicle to the dock phase for another attempt within the same mission.
The edge exists but rarely fires: 0, 1 and 2 missions used it in the three arms of Section 6,
and 2 of the handoff run's docks re-closed. Where it fires, the recorded grasp depth is that of
the final closure, so the handoff analysed downstream is the one the mission actually flew with.

Failing an acceptance test does not by itself end a mission. The vehicle stays in the current
phase and continues until that phase's budget is exhausted, at which point the mission is
recorded as a failure of that phase. The one exception is the gating arm of Section 6, in which
a rejected handoff ends the mission and is recorded separately as a rejection.

The dock phase is analytical rather than learned because a learned policy for the same
descent-and-grasp proved far less reliable in earlier development; the analytical controller
that replaced it commits 383 of 384 baseline missions.

### 3.3 The acceptance test and the instrumented interface

Each phase boundary is guarded by an acceptance test evaluated on the vehicle state and written
as a predicate. At the interface,

*A* = 1[*c* ≥ 200] ∧ 1[‖*p* − *p*_box‖ < 0.30 m] ∧ 1[*θ* < 15°] ∧ 1[‖*ω*‖ < 2 rad/s]  (1)

where 1[·] is 1 when its condition holds and 0 otherwise, *c* is a cumulative count of control
steps on which more than half of the payload's horizontal cross-section lies inside the
rectangle spanned by the jaw struts while the payload is within their vertical range, *p* is the
vehicle position, *θ* the tilt from vertical and *ω* the body rate. Predicate (1) is a
containment-and-attitude test: it asks whether the payload was under the jaws for long enough
and whether the vehicle is level and slow. It does not ask what the jaws did to the payload. The
tests at the other two boundaries are given in Figure 2. The approach records zero losses (Figure 2)
not because it produces no bad states but because its test cannot refuse: both remaining
boundaries are permissive rather than strict, each admitting a mission on a timeout fallback as
well. The containment counter *c* recurs throughout the paper. The close command itself is
issued at *c* = 225; the baseline
pipeline commits at *c* ≥ 200 (1.33 s of cumulative containment at 150 Hz), the instrumented
runs at *c* ≥ 325, the shaped closure of Section 6 ramps between 225 and 290, and its gate tests
from 200 and refuses at 400. One consequence should be stated plainly: in the baseline arms the
commit precedes the close command by 25 contained steps, so the closure completes in the first
half-second of the climb phase; in the instrumented runs, which commit at 325, it completes
before the handoff. Every stratified result of Section 5 uses the instrumented runs, where the
grasp exists at the boundary it is measured at.

Section 3.4 introduces two instrumented runs, and both commit on the simpler rule the project's
state-capture tooling had used from the start: a longer containment count together with the
distance term, *c* ≥ 325 ∧ ‖*p* − *p*_box‖ < 0.30 m, and no attitude test at all. That
difference does two things. It is why 72 of the 383 handoffs analysed in Section 5.2 arrive with
a tilt or a body rate that (1) would have refused. It is also what makes the tilt analysis of
Section 5.4 possible. Where a result depends on which form was in force, the text says so.

**[Figure 2 here]**

*Figure 2. Mission chain, the acceptance test at each phase boundary, and the grasp depth s.
Numbers above the arrows are missions crossing that boundary; numbers below each phase are
missions lost in it, from the baseline run of Section 3.4 (N = 384). Four of the 49 climb-phase
losses occur in the interval in which the vehicle is held steady after the climb. The approach
and climb-exit tests also admit a mission on a timeout fallback. The grasp depth is fixed when
the jaws are stopped by the payload and is then carried onward unchanged; it appears in none of
the tests.*

Separately from the acceptance tests, I instrumented the interface. Each quantity is defined
once here and is drawn in the figure where it acts: the clearance and the jaw geometry in
Figure 1(c) and (d), the closure quantities on the axes of Figure 3. Let *R* be the vehicle
attitude and *p*_box the payload centre. The payload position in the vehicle body frame is

*d* = *R*ᵀ (*p*_box − *p*),  *d* = (*d*_x, *d*_y, *d*_z)  (2)

and the three readings used in this paper are components of that one vector, taken at two
instants. At the moment the grasp is handed over, the *grasp depth*

*s* = |*d*_y|  (3)

is the distance by which the payload centre lies off the vehicle centreline along the closing
axis. The payload does not move itself: it is displaced by the one jaw that keeps closing after
contact, so *s* measures the jaw's advance after the payload begins to resist the closure, and a
larger *s* means a deeper grasp. The name is fixed by that correspondence, which Section 5.1
measures directly, and not by any depth of the payload inside the gripper. Read at the close
command instead, the same vector gives the *strut-axis misalignment* *e*_x = *d*_x and the
*initial offset* *s*₀ = |*d*_y|, and *d*_z combined with the hinge height of Section 3.1 gives
the *vertical clearance* *h*.

Of all these quantities, only *c*, *θ* and *ω* appear in (1); *d* and everything derived
from it appear in no acceptance test at all. Note what this does and does not mean: computing
*c* already requires the payload's pose, so the pipeline possesses the information from which
*s* could be read at every step. What is missing is not sensing but a test — the information is
collapsed into a boolean containment count, and *s* is never formed.

A grasp driven far enough for the foot to reach the payload is called *engaged*. Section 5.1
fixes the working threshold on *s* and measures how closely it corresponds to actual foot
contact.

### 3.4 Evaluation runs

All results come from one configuration of the Isaac Lab GPU-parallel robot learning framework
(version 0.39.0), the successor of the Orbit framework [12], evaluated in 128 parallel
environments. Per-mission randomization covers the vehicle mass scale (0.9–1.1), the motor
thrust-coefficient scale (0.85–1.15) and the starting angle of a slowly varying wind
disturbance, in the manner of dynamics randomization [13]; the payload mass, its position on the
pedestal, the contact friction and the centre-of-mass offset are held constant. Every launched
mission is run to its end, whether it succeeds or fails, so no mission is truncated when a
target count is reached.

Three runs of this configuration are used. Table 2 gives their sizes, and no figure or table
mixes them.

*Table 2. The three evaluation runs. The closure run yields two sample sizes because one mission
— a dock timeout — closed its jaws but was lost before a final payload pose could be logged;
jaw-angle statistics use 297 grasps and every statistic involving s uses 296.*

| Run | Launched | Lost in dock | Accepted at the interface | Reached delivery | Seed | Records | Used for |
|---|---|---|---|---|---|---|---|
| Baseline | 384 | 1 | 383 | 334 | 1 (fixed) | per-mission outcome | Figure 2; Section 6 |
| Handoff | 400 | 17 | 383 | 341 | not fixed | state at climb and delivery entry | Sections 5.2, 5.4 |
| Closure | 300 | 4 | 296 | 275 | not fixed | full closure traces and mission outcomes | Sections 5.1, 5.3 |

The two instrumented runs differ from the baseline in two further respects, and both are visible
in Table 2. They lose more missions inside the dock phase, partly because their commit rule
demands a longer containment count, which leaves more time in the dock phase to fail. And they
pass the vehicle from the climb straight to the delivery policy, without the stabilization
interval the baseline runs, so the four baseline losses recorded in that interval have no
counterpart in them and the two runs enter the delivery phase at slightly different instants.
The runs also differ in the quantity this paper is about, the engaged share being 57.4% in the
handoff run against 49.3% in the closure run (Fisher *p* ≈ 0.03) — an unexplained difference
between two unseeded runs of nominally the same configuration. Two runs cannot estimate a
variance, so this gap is treated as a caution against comparing rates across runs, not as a
noise floor; the arm comparisons of Section 6 share one fixed seed and are a separate matter.

Downstream of the dock the runs agree closely: the handoff run survives the climb in 89.0% of
accepted handoffs against 87.2% in the baseline, and completes 70.2% of accepted handoffs
against 69.2%. All stratified results below use accepted handoffs only, so the dock-phase
difference does not enter them.

---

## 4. Hypotheses

**H1.** The mission losses are governed not by the competence of the downstream phases alone but
by the grasp depth *s*, which is produced by the closure in the dock phase and inherited by the
phases that follow.

**H2.** Whether a grasp ends engaged is accounted for by the configuration at the instant the
closure is commanded, through the three quantities defined in Section 3.3:

*E* = *f* (|*e*_x|, *h*, *s*₀)  (4)

Here *E* is 1 for an engaged grasp and 0 otherwise, and only the magnitude of *e*_x is used. The
first of the three carries a threshold predicted by the geometry rather than fitted to the data:
the inner faces of the jaw members sit 44 mm from the vehicle centre and the payload half-width
is 40 mm, leaving 4 mm of misalignment before a member can meet the payload (Figure 1(d)). The
prediction *e*\* = 4 mm is therefore fixed before any grasp is measured. No threshold is
predicted for *h* or for *s*₀; those terms are empirical. Section 5.3 traces part of the
clearance term's mechanism to arrival momentum.

Section 6 evaluates the three interventions reachable without retuning or retraining any policy:
shaping the closing motion, gating the handoff on measured depth, and re-grasping when the depth
test fails. All three act once the close command has been issued, late relative to the contact
that fixes the quantity; what each of them can and cannot show is stated with its result. Moving
the terms of (4) themselves — the setpoint experiment of Section 7 — requires retuning the dock
controller and is not evaluated here. Field practice supplies the natural point of comparison:
the MBZIRC 2020 wall-building winner aborts a grasp *during* the interaction, before committing
to lift — when its attitude error shows the vehicle bearing on the ground through its landing
gear, when the brick is gripped far from its centre of mass, or when the estimated mass shows no
brick attached at all [2].

---

## 5. Measurements

### 5.1 What the closure does

The feet, closed, span less than the payload (Figure 1(c)), so the jaws cannot reach the pose
they are commanded to. The commanded −5° is in fact beyond the mechanism's own reach even
without a payload — at that pose the two feet would interpenetrate by 33 mm — and is used as a
command past the kinematic limit so that the actuators keep applying closing force wherever the
jaws are stopped; self-collision between the jaws is disabled in the model. Across the closure
run the jaws reached −5° in none of the 297 recorded closures. What they do instead is not a
smaller version of what was asked for.

Figure 3(a) follows both jaws and the payload through the closure. For the first part of the
motion the jaws behave exactly as commanded. Their angles stay within 1° of each other down to a
median 31.2° from vertical (interquartile range 27.9–38.7°). On the way down they already brush
the payload — lightly enough that both jaws still track their command — and this first contact
centres it: the payload arrives at the close command a median 11.4 mm off the centreline and
sits within 1.0 mm of centre at the moment the two jaw curves first separate (IQR 0.5–1.5 mm),
which is the moment its resistance becomes strong enough to arrest the motion. Section 5.3 gives
the full distribution of that initial offset.

Then the symmetry fails. In every one of the 297 recorded closures one jaw drives on past the
other — how far, varying from grasp to grasp, is the quantity at issue — while the other is
forced back outward, reaching its 50° open limit in 292 of them. At the median the advancing jaw
ends at 14.7° and the payload finishes 24 mm off the centreline, but both figures are medians
over the two very different outcomes that the rest of this section separates. Nothing in the
model asks for this. The two jaws share one commanded angle, one actuator specification and
mirrored hinge axes, and the command is still symmetric while the outcome is not. Two jaws
squeezing a body wider than their closed span cannot both advance, and the symmetric
configuration is the one arrangement in which neither has won. Once one jaw is marginally ahead,
the payload moves towards the other; that lengthens the leading jaw's moment arm — the
perpendicular distance from its hinge to the contact force — while the payload, moving toward
the trailing jaw, bears on its inner face and forces it back open against its position command.
The process has no observed return: symmetry is not recovered in any of the 297 grasps.

Call the jaw that does advance the winner. If a physical perturbation selected the winner, the
payload's offset at the onset of arrest should decide it: a payload sitting toward one jaw
should stall that jaw and hand the win to the other. The data refuse this. Just before the jaw
curves separate the offset's sign is nearly balanced — positive in 135 of the 297 closures,
negative in 162 — yet the same jaw won every one of the 297, including all 135 in which the
perturbation pointed the wrong way. The winner is therefore not selected by the payload; and
since the vehicle description itself is mirror-symmetric — the two jaws' masses, inertias,
collision geometry and joint parameters are identical up to sign, which I verified in the model
file — the selection can only live in the contact solver's deterministic tie-breaking. Section 7
states what does and does not survive that fact. In five of the 297 closures the losing jaw was
not driven fully back, ending between 15° and 39° rather than at its limit.

Two different contacts can bring a jaw to rest, and telling them apart matters for Section 5.3.
The first is the one just described: a foot meets the payload along the closing axis. The
payload can slide ahead of a foot, so the jaw keeps going until it can push the payload no
further. The second is a jaw member meeting the payload along the strut axis. A member meets the
payload's side face head-on rather than sliding along it, so it cannot push the payload aside; a
jaw that fouls one stops early and its foot never arrives. The 4 mm figure of Figure 1(d) is the
misalignment above which the second kind of contact becomes geometrically possible, and Section
5.3 tests whether it matters.

The angle at which the advancing jaw stopped — its *arrest angle*, *β* — ranged from 6.0° to
42.2°, and *β* and the grasp depth are two readings of one event: their Spearman correlation is
−0.91 over the 296 grasps. Reconstructing the foot-to-payload distance from the part geometry
separates the two outcomes cleanly. Counting a gap below 2 mm as contact, which allows for the
error of the reconstruction, 97.9% of grasps with *s* > 25 mm have at least one foot touching
the payload and 97.9% of grasps with *s* < 15 mm have none. The median gaps are −3 mm against
+17 mm; a negative gap means the reconstructed bodies overlap, which in this contact model is
what load-bearing contact looks like. Under the stricter criterion of a gap below 0 mm the first
figure falls to 93.2% and the second does not move. Median arrest angles are 7.5° and 32.5°.
Those two cuts name the two ends of the distribution and are used wherever a downstream outcome
is stratified: *engaged* means *s* > 25 mm, *not engaged* means *s* < 15 mm, and the band
between them is reported separately. Section 5.3, which predicts engagement rather than an
outcome beyond it, uses the single cut *s* > 25 mm over the whole sample. They bracket the
trough of the closure run's own bimodal distribution of *s*, and the contact statistics are what
they rest on.

**[Figure 3 here]**

*Figure 3. Closure run. (a) Both jaw angles and the payload offset through the closure: median
curves with interquartile bands, over the 297 grasps whose closure was recorded from the close
command onward. Individual grasps break symmetry at different steps, so the median curve is not
the curve of any one grasp and the offset it shows at the dashed line, 2.1 mm, is larger than
the per-grasp median of 1.0 mm. (b) Grasp depth against arrest angle, over the 296 grasps with a
recorded final payload pose; the engaged and not-engaged configurations themselves are drawn in
Figure 1(c).*

The grasp depth is therefore a continuous record of the advancing jaw's post-contact drive, and
the 25 mm threshold marks the drive at which its foot reaches the payload. Because the closure
is one-sided, an engaged grasp is not an enveloping one; the opposing jaw ends at its open limit
and holds nothing, and the payload cannot reach the vehicle body, which sits 74 mm above its top
face. Working the median engaged configuration through the part geometry gives exactly two
contacts, both on the advancing side: the foot's upper face beneath the payload's bottom (a
shelf), and the jaw member against the payload's near side face below its top edge (a wall) — an
L-shaped grip from a single jaw. In a not-engaged grasp the foot stops short and the side
contact alone is left to hold the payload by friction. Contacts of that second kind carry far
less load in one comparable aerial gripper, whose fully enveloping grasps held 2 kg while its
pinching grasps failed consistently once one test object reached 250 g [4]; the comparison
bounds the pinching side only, since an engaged grasp here is not an enveloping one either. The
evidence here is the correspondence just given together with the failure modes of Section 5.2;
the contact model is the simulator default and the friction is held constant, so no friction
sweep supports it, and it does not cover every case, since half of the not-engaged grasps
complete their mission. Stratifying those survivors against the droppers — within the closure
run's own recorded outcomes — gives one clue and one surprise: neither alignment, nor clearance,
nor the depth itself separates the two groups, but the vehicle's tilt at commit does (median
9.4° among survivors against 2.6° among droppers, Mann-Whitney p < 0.001) — the flattest,
cleanest-looking shallow grasps are the ones that drop. These data do not explain the direction;
the mechanism that retains a shallow grasp remains open.

One practical note closes this section. Equation (2) needs the payload pose, which a real
vehicle does not have. The arrest angle needs only a joint encoder, is available on any actuated
gripper, and stands in for the grasp depth at ρ = −0.91. An onboard version of the measurement
proposed here would read *β* rather than (3).

### 5.2 The losses are inherited, not generated where they are recorded

I took the 383 accepted handoffs of the handoff run — accepted, in this run, on the reduced
criterion of Section 3.3 — and sorted them by the grasp depth *s* measured at the interface:
engaged (*s* > 25 mm, *n* = 220), not engaged (*s* < 15 mm, *n* = 108), and the remainder mid
(*n* = 55). The three-way split is used for the outcome table below and for the tilt
stratification of Section 5.4, where the mechanical distinction of Section 5.1 is cleanest at
the two ends.

**[Figure 4 here]**

*Figure 4. Handoff run. (a) Grasp depth measured after acceptance: the dock acceptance test
admitted the full spread. (b) Three rates for the engaged and not-engaged groups — climb passed,
delivery phase succeeded, and mission completed — with Wilson 95% intervals [14]. The mid group
is omitted from (b) and reported in Table 3; the reversed ordering of the climb pair is a tilt
effect, resolved in Table 4 (Section 5.4). The delivery-phase rate is conditional on surviving
the climb (n = 191 and 102).*

The engaged and not-engaged groups differ by 28.2 percentage points in mission completion
(Figure 4), and the difference is not made in the climb. The not-engaged group in fact survives the climb more
often, 94.4% against 86.8% — an ordering that appears to run against H1 and that Section 5.4
resolves as an effect of tilt rather than of depth. The separation appears in the delivery phase
instead, 91.1% against 53.9%. Mission completion falls monotonically across the three groups, so
the middle band behaves as its position on the scale predicts.

Table 3 shows how the missions failed. The failure that dominates the not-engaged group is the
payload being lost in flight, at 5.3 times the rate of the engaged group — the failure mode the
retention argument of Section 5.1 predicts. The tilt-limit column moves with it: 19.4% of
not-engaged handoffs end the delivery phase past the 70° threshold, against 3.2% of engaged, so
a shallow grasp costs attitude as well as retention.

*Table 3. Outcome of accepted handoffs by grasp-depth group (handoff run, 383 handoffs).
Percentages are of the group and sum to 100%: among accepted handoffs no mission of this run
failed by timeout or by excessive distance, and every loss falls in one of the three loss
columns. A payload loss is counted in the phase in which it occurred, so the two delivery
columns exclude losses during the climb.*

| Group | *n* | Median *s* | Completed | Payload lost in delivery | Tilt > 70° in delivery | Lost in climb |
|---|---|---|---|---|---|---|
| Engaged (*s* > 25 mm) | 220 | 35.8 mm | 79.1% | 4.5% | 3.2% | 13.2% |
| Mid (15–25 mm) | 55 | 20.2 mm | 72.7% | 3.6% | 10.9% | 12.7% |
| Not engaged (*s* < 15 mm) | 108 | 9.1 mm | 50.9% | 24.1% | 19.4% | 5.6% |

The same gradient is present over all 383 handoffs with no grouping at all: completion rises
with *s* — point-biserial *r* = 0.29 (*p* = 8 × 10⁻⁹), odds ratio 1.69 per 10 mm, and an area
under the ROC curve (AUC) of 0.690, where 0.5 would be chance and 1.0 perfect separation — and a
plain median split at 30.6 mm gives 80.1% against 60.4%. Nor is the gradient only the binary
contact state in disguise: within the engaged group alone, deeper grasps still complete more
(*r* = +0.20, *p* = 0.003 over 220), while within the not-engaged group there is no gradient
(*r* = +0.03). The 28.2-point figure is the contrast
between the two ends of that gradient, not a discontinuity the grouping created.

Three statements follow. First, the climb phase does not degrade with poor grasp depth. Second,
the quantity that predicts the delivery outcome is fixed at the closure, two phases earlier.
Third, the aggregate delivery-phase rate is a mixture of those two populations and is therefore
not a property of the delivery policy alone.

The third statement admits an alternative reading. The delivery policy was not trained on the pipeline's own output. It was trained from a
bank of 4996 grasp states pre-generated by running the same dock and climb controllers and
capturing the state of every attempt that reached climb altitude with the payload still held,
re-docking after a failed closure until one succeeded. That bank is 99.2% engaged with a mean
grasp depth of 45.8 mm, whereas the live pipeline hands the policy a mean of 25.8 mm, with 29.9%
of arrivals not engaged, over the 341 missions that reach the delivery entry. The capture
criterion does not account for the difference — Section 5.4 shows climb survival barely responds
to grasp depth — but the generator's entry conditions do: it skips the approach phase entirely,
spawning the vehicle 0.30–0.50 m directly above the payload within ±8 cm and ±0.2 m/s, an entry
distribution far tighter than the approach policy delivers. That a settled entry deepens the
grasp is exactly the direction Section 5.3 measures on the live runs.

A policy evaluated on states absent from its training distribution degrades, and the degradation
compounds along a trajectory [10]. The delivery failures of the not-engaged group therefore mix
a physical component with a training-coverage component, and these data cannot separate them;
Section 5.4 gives the coverage component a concrete candidate form. One part of the finding
survives either way: the quantity that sorts the two populations is set two phases upstream, and
no test in the pipeline looks at it. Both components are hidden for the same reason.

The quantity has one further property, and it is what makes that concealment costly. Writing
*s*ₖ for the depth measured at the climb entry and *s*ₖ₊₁ for the same quantity at the delivery
entry, over the 341 missions that reach delivery,

*s*ₖ₊₁ = *s*ₖ + *ε*,  *r* = 0.9994,  mean |*ε*| = 0.055 mm  (5)

where *ε* is the per-mission change. The climb carries the grasp depth forward without altering
it; the largest single change observed is 8.2 mm, and exactly one of the 341 missions crosses a
group boundary between the two measurements. This is what makes the quantity worth measuring at
the interface rather than later: a phase handed a shallow grasp has nothing left to recover
with. The delivery phase is not instrumented at its exit, so the claim of invariance covers one phase and
not two. H1 is supported.

### 5.3 What accounts for the engagement

As set out in Section 5.1, this section splits the 296 closure-run grasps in two at *s* = 25 mm
— engaged against everything else — rather than using the three-way split of Section 5.2.

The alignment threshold predicted in Section 4 holds: grasps with |*e*_x| < 4 mm engage in 90.5%
of cases (95% interval 78–96, *n* = 42) against 42.5% for the rest (37–49, *n* = 254), Fisher
exact *p* = 2 × 10⁻⁹. Fitting the cut to the data instead lands at 5.5 mm (rate gap
89.6% against 35.2%), within 1.5 mm of the predicted value.

**[Figure 5 here]**

*Figure 5. Closure run, n = 296, all three quantities measured at the close command. (a) Every
grasp as a scatter of strut-axis misalignment |e_x| against vertical clearance h, outcome shown
by marker; the dashed line is the 4 mm interference threshold predicted by the jaw geometry
before the data were examined. (b) Engagement rate over terciles of h (horizontal) and |e_x|
(vertical, increasing upward), with cell counts below each rate. (c) Engagement rate over
terciles of the initial offset s₀, with Wilson 95% intervals [14].*

Taken on its own, the vertical clearance *h* — how high the jaw hinges sat above the payload
when the jaws were told to shut — separates the outcomes only weakly, and its relation to
engagement is not monotone until the other terms are held fixed. Its mechanism, however, is
measurable in these logs: a jaw released from higher up is moving faster when the payload begins
to resist, and the advancing jaw's angular speed at that moment correlates with the final depth
at *r* = +0.49 — more strongly than *h* itself (*r* = +0.42), with *h* feeding the speed at *r*
= +0.26. Clearance appears to act through arrival momentum: a longer swing arrives harder and
drives deeper. The non-monotonicity is an artefact of a confound: at the close command the 34
grasps with the smallest clearance (*h* < 38 mm, the lowest 11.5%) are also the best aligned:
their median |*e*_x| is 4.5 mm against a median of 13.4 mm for the other 262. Stratifying by
both variables resolves it, and the joint structure is visible in Figure 5(b). When the
misalignment is below 7 mm the grasp engages in 68% to 82% of cases across the three clearance
cells; when it exceeds 17 mm the grasp engages only when the clearance exceeds 49 mm, where the
rate is 73.5% (*n* = 49) against 0% below that clearance (*n* = 50). Unlike *e*\*, the 4 mm
geometric tolerance of Section 4, the 49 mm figure is the upper tercile boundary of the observed
distribution, so it is empirical and depends on the binning.

The third term of (4) is the offset the payload already has when the jaws start to move — though
Section 5.1 showed the closure erases exactly this offset before the arrest begins, which is
what makes its predictive power surprising. It ranges from 0.1 to 55.7 mm with a median of 11.4
mm, and it is the strongest single predictor of the three. Note that the two offsets point
opposite ways: misalignment across the struts, *e*_x, hurts engagement, while offset along the
closing axis, *s*₀, helps it. Over its terciles, cut at 8 and 16 mm, the engagement rate rises
21.2%, 43.9%, 82.8% (*n* = 99, 98, 99), monotonically and with an AUC of 0.798, against 0.699
for
|*e*_x| and 0.637 for *h*. It also settles how much of the grasp depth is inherited and how much
is made: *s*₀ accounts for 32% of the variance of *s* (*r* = 0.567), so most of the variation in
the final depth arises during the closure rather than being carried in from the approach.

How *s*₀ predicts is less obvious than that it does. Section 5.1 showed the two jaws erasing the
initial offset before either of them wins: whatever offset the payload starts with, it is
centred to about a millimetre by the onset of arrest. So *s*₀ cannot be acting through where the
payload sits at contact. It is nearly orthogonal to the strut-axis misalignment (*r* = 0.02) but
strongly correlated with the vertical clearance (*r* = 0.68), and a vehicle that stops high and
off-centre is one whose descent has not settled. On this evidence *s*₀ and *h* are two readings
of the same unsettled approach, and these data cannot say which of them the mechanism responds
to.

A logistic classifier on |*e*_x| and *h* reaches an out-of-fold AUC of 0.848 under stratified
5-fold cross-validation [15]; adding *s*₀ raises it to 0.883. On 296 grasps that increment does
not separate the two models, and nothing here depends on separating them: the point of (4) is
that three quantities available at the close command account for an outcome the pipeline never
checks.

The mean strut-axis misalignment of this dock controller is 13.4 mm, more than three times the 4
mm the jaw geometry allows, so tightening the tracking loop is one design direction for this
platform. It is not the only direction. The clearance and initial-offset terms both recover part of the misaligned
population without touching the tracking loop, as the high-misalignment row of Figure 5(b) shows
— though by the correlation above those two may be readings of one underlying condition, the
unsettled approach, rather than two independent levers. What the measurement establishes is
three measured quantities and at least two distinct causes; which is cheapest to move is a
platform question. H2 is supported — its first term by a prediction the data then confirmed, its
other two empirically.

### 5.4 Which quantity governs which phase

Grasp depth is not the only thing that crosses the interface. The vehicle also arrives with an
attitude, and predicate (1) has a term for it. Separating the two is possible in the handoff run
precisely because that run committed on the attitude-free criterion of Section 3.3: 57 of its
383 accepted handoffs carry a tilt of 15° or more, which the full predicate would have refused.
That population does not exist in the pipeline of Figure 2, so the tilt results below describe a
variant of it with the attitude term switched off. High-tilt handoffs that reach the delivery
phase do so through the climb-exit timeout fallback, since this run has no stabilization
interval.

*Table 4. Loss rate by grasp-depth group and tilt at the interface (handoff run), for the climb
phase and, conditional on surviving it, for the delivery phase. All eight cells of the
stratification are shown, with counts; p-values in the text are Fisher exact, uncorrected.*

| Loss rate | Climb, tilt < 15° | Climb, tilt ≥ 15° | Delivery, tilt < 15° | Delivery, tilt ≥ 15° |
|---|---|---|---|---|
| Engaged | 5.3% (*n* = 187) | 57.6% (*n* = 33) | 7.3% (*n* = 177) | 28.6% (*n* = 14) |
| Not engaged | 3.1% (*n* = 97) | 27.3% (*n* = 11) | 50.0% (*n* = 94) | 0% (*n* = 8) |

Read column by column, the table gives each quantity a first-order home and both quantities a
second-order reach into the other's phase. Tilt dominates the climb: within the engaged group it
raises the climb loss from 5.3% to 57.6% (*p* = 7 × 10⁻¹²). Depth dominates the delivery phase:
within low tilt, a shallow grasp loses its payload at 50.0% (95% interval 40–60) against 7.3%
(4–12) for an engaged one (*p* = 3 × 10⁻¹⁵). But the second-order cells are not flat. Under high
tilt in the climb, engaged grasps die at twice the not-engaged rate, 57.6% against
27.3% — an underpowered contrast (*p* = 0.16 on 33 and 11 handoffs), not an absent one. And in
the high-tilt column of the delivery phase the shallow grasps that should be dropping do not: 0
of 8 are lost, against 50.0% of the 94 low-tilt shallow grasps (*p* = 0.007). Tilt at the
interface protects a shallow grasp in flight — the same regularity Section 5.1 found
independently in the closure run's survivor tilts, and in neither run is the direction
explained.

These cells change the reading of Section 5.2's reversed climb ordering. Decomposing the
7.6-point gap (13.2% against 5.6%) over the strata: 1.8 points come from composition — high-tilt
handoffs are more frequent among engaged grasps (15.0%, against 23.6% in the mid band and 10.2%
among the not engaged, a non-monotone pattern) — 2.0 points from the low-tilt rate difference,
and 3.8 points, half the gap, from the high-tilt column where engaged grasps die at double the
rate. Composition is a quarter of the story, not the whole of it.

The half that composition does not explain has a candidate mechanism inside the definition of
*s* itself. The grasp depth is the payload's distance off the vehicle centreline, so a deep
grasp is also an eccentric load: at the engaged median of 35.8 mm, the 0.20 kg payload applies a
0.070 N m roll moment and shifts the combined centre of mass 5.4 mm off axis. Depth and
eccentricity are the same number read twice, and they pull in opposite directions — depth
secures the payload for the delivery phase, eccentricity loads the attitude loop during the
climb, exactly where the high-tilt engaged cell shows the excess deaths. The same duality gives
the training-coverage component of Section 5.2 a concrete form: a delivery policy trained at a
mean eccentricity of 45.8 mm and handed 25.8 mm arrives roll-trimmed for a load it is not
carrying. These logs record no commanded attitude, so the account remains a hypothesis; one
logged run of the delivery policy's roll commands against *s* would test it directly.

Tilt is in predicate (1); grasp depth is not. Tilt losses are what a threshold set too loosely
costs, and tightening a term the pipeline already evaluates would reduce them. Grasp-depth
losses cannot be reached that way, because there is no term to tighten. Both are invisible in a
per-phase rate; only one is invisible to the pipeline itself.

---

## 6. Interventions at the Interface

I evaluated the three no-retrain interventions of Section 4 with the same seed as the baseline
arm. Shaped closing ramps the jaw command linearly from the open pose to the closed pose between
containment counts *c* = 225 and *c* = 290 — at least 0.43 s at 150 Hz, and longer whenever
containment lapses, so the ramp's duration varies with a quantity correlated with the outcome; a
wall-clock ramp would remove that flaw. Handoff gating adds a depth term to predicate (1) and
rejects handoffs that fail it. The term as implemented tests the signed offset *d*_y rather than
its magnitude, which is equivalent here because the same jaw advances in every grasp of Section
5.1. Re-grasping applies the same test after the closure and, on failure, reopens the jaws and
re-docks, up to three attempts — an extension of the pipeline's existing retry edge (Section
3.2) from "the payload was left behind" to "the payload is seated too shallow". Shaping and
gating were evaluated together in one arm, so their individual contributions are not separated.

*Table 5. Outcome of the intervention arms; the re-grasp arm used half the sample size of the
others. Shares are percentages of all missions of that arm; conditional completion is over
missions whose handoff was accepted, which for the arms without a gate means missions that did
not fail inside the dock phase.*

| Arm | *N* | Completed | Rejected | Lost in dock | Lost in climb | Lost in delivery | Completed \| accepted |
|---|---|---|---|---|---|---|---|
| Baseline | 384 | 69.0% | — | 0.3% | 12.8% | 18.0% | 69.2% |
| Shaped close + gate | 384 | 38.0% | 52.1% | 4.9% | 3.1% | 1.8% | 88.5% |
| Re-grasp (up to 3 attempts) | 192 | 63.0% | — | 16.1% | 10.4% | 10.4% | 75.2% |

**[Figure 6 here]**

*Figure 6. (a) Outcome composition of each arm; the dashed line marks the baseline's 69.0%
completion. (b) Share of all missions of the arm lost in the dock and delivery phases;
rejections appear in (a) and are excluded here, so the gate arm's delivery share is depressed by
the missions it never admitted. (c) Throughput against conditional reliability, with Wilson 95%
intervals on the conditional rate [14]. All arms use the baseline run's seed.*

Figure 6 shows where each arm's missions ended. In the delivery phase each arm did what it was
designed to do. The delivery-phase loss fell from 18.0% of missions to 1.8% under gating and to
10.4% under re-grasping (*p* = 0.020 for the latter, uncorrected for the several comparisons
this table invites). Neither raised mission completion. Under gating part of the fall is a
denominator effect, since 52.1% of missions never entered the delivery phase at all; computed
over accepted missions only, the rate rose from 69.2% to 88.5%. The losses that left the
delivery phase reappeared upstream. Dock-phase loss rose from 0.3% to 4.9% in the gate arm and
to 16.1% in the re-grasp arm, and of the 52.1% the gate refused, 46.4 percentage points were
refused by the new grasp-depth term and 5.7 by the tilt term predicate (1) already carried.
Climb loss also fell under the gate, from 12.8% to 3.1% — the same movement again, failures
leaving a downstream phase for the rejection column.

Re-grasping is the clearest case. It reduced the delivery loss significantly yet completed 63.0%
against 69.0%, a difference that is not significant (*p* = 0.16), because reopening the jaws
lost the payload during the dock in 9 of 192 missions and exhausted the retry budget or the
phase timeout in a further 22.

The arms buy certainty about the missions they accept. Call P(completed) the throughput and
P(completed | handoff accepted) the conditional reliability. Across the three arms of Table 5,
reliability rises as throughput falls, and no arm improves on both.

None of the arms tests whether a late intervention could ever raise completion. Gating cannot by construction: a rejected handoff
ends the mission (Section 3.2), so the gate term can only subtract completed missions, never add
them. Re-grasping does not act on a fixed grasp at all — it discards it and repeats the whole
closure. Shaped closing, the one intervention that reaches into the closure itself, shared an
arm with the gate, so its own contribution to completion cannot be read off Table 5. The table's
finding is therefore the trade itself, not a verdict on that question.

Nor can shaped closing's effect on the grasp depth be read off the instrumentation. The logs
record the depth at a fixed containment count, which lands after the jaws have stopped when the
command is a single step but part-way through the closure when the command is a ramp ending at
*c* = 290. The gate provides the only other reading, and it is biased the other way: it tests
the depth at every step from *c* = 200 and holds a mission in the dock phase until *c* = 400
before refusing it, giving the shaped arm a longer window to satisfy it than the baseline ever
receives. No third measurement point exists in these runs, so the question stays open.

Without that measurement, the experiment that remains is shaped closing run alone, on enough
missions to exclude a several-point gain. That is the one probe that reaches
inside the closure, and by the arrival-speed result of Section 5.3 it carries a directional
prediction: a slower ramp should arrive with less momentum and produce shallower grasps.

---

## 7. Discussion

**What the measurements support.** The losses of this system are governed by a quantity produced
by a contact event, carried through the following phase without change, and absent from every
acceptance test in the pipeline. Because that quantity is not measured, the per-phase rates
cannot be read as component properties: within the handoff run the delivery phase completes
78.9% of the missions it receives, while running at the two rates of Section 5.2 on the two
populations it is handed. How much of that gap is physical and how much is the delivery policy
having never been trained on the second population is not resolved by these data (Section 5.2);
the point that does not depend on the split is that neither part is visible in the per-phase
rate.

The pattern itself is not particular to this gripper. Any pipeline whose boundary test is
boolean while the contact that precedes it fixes a continuous, uninspected quantity has the same
exposure. One field system of Section 1 certified a grasp its gripper had not made [1]; the
other's grasp-time safeguard — watching estimated mass and attitude during the interaction [2] —
is exactly a continuous interface measurement retrofitted onto a boolean gate. A gripper
correctly sized for its payload would remove this paper's particular quantity; it would not
remove the class.

**Design consequences.** Four follow. First, measure a continuous quality at the boundary
alongside the boolean acceptance test. Second, check a candidate quantity for downstream
invariance before investing in the phases that receive it: a quantity that behaves as (5) does
across the phase where it was measured is unlikely to be repaired by a later one, so anything to
be done about the quantity itself should be done at or before the boundary. Third, treat its
consequences as a separate problem, because they can still be reduced downstream by a receiving
policy trained to tolerate poor values; the grasp depth therefore belongs among the
training-distribution variables of the receiving policy and not only among the properties of the
phase that produces it. Fourth, place a quality check as early in the closure timeline as the
sensing allows, as [2] does; on hardware that check would read the arrest angle *β* rather than
(3), for the reason given in Section 5.1.

**Limitations.** The results are from simulation, one gripper and one payload, and the mission
ends with the payload still held: no release or placement is modelled, although setting a
shallow grasp down accurately is precisely where its depth should matter next. The 4 mm and 49
mm thresholds are properties of that pair; a compliant gripper or a closed geometry matched to
the payload would widen them. The payload position on the pedestal is not randomized, so the
misalignment distribution of Figure 5(a) is produced by the approach and by wind and mass
dispersion only, and would be wider under realistic placement uncertainty. The two learned
policies enter this paper as fixed artefacts: their training recipes are not part of the claims
and are not reproduced here. The intervention arms use a single seed and the re-grasp arm has
half the sample of the others, so the comparisons between arms rest on one realization of the
randomization; the unexplained between-run difference of Section 3.4 counsels against reading
any of them across runs.

One limitation outweighs the rest. The symmetry breaking of Section 5.1 is the mechanism on
which the whole account rests, and it has been observed in one contact model. The sign analysis of Section 5.1 already places
the winner selection in the solver's deterministic tie-breaking, a property of the model; what
no analysis of these logs can settle is whether the instability itself, and with it the spread of *s*, survives under
a different contact model. Until a replication answers that, the distribution of grasp depths
reported here should be read as a property of this model rather than of the mechanism.

**Next steps.** Three experiments follow. The first moves a term of (4) directly, without
retraining any policy: raising the height at which the dock controller stops its descent. The
stratification of Figure 5 suggests a large effect, but raising the setpoint also changes the
descent dynamics, so the observational strata are a hypothesis rather than a prediction. The
second rebuilds the delivery training bank from the handoff distribution the pipeline actually
produces and fine-tunes the delivery policy, which would separate the physical and
training-coverage components of Section 5.2 and would settle the question left open there. The
third repeats the closure run under a different contact solver, to establish whether the
instability of Section 5.1 is a property of the mechanism or of this model.

---

## 8. Conclusion

I measured a complete aerial pick-and-deliver pipeline end to end and traced its losses to their
origin. They are not made in the phase that records them. They are carried by the grasp depth,
fixed when a payload too wide to enclose brings the jaws to rest, untouched by the phase that
carries it onward, and read by no acceptance test in the pipeline. Sorted by it, handoffs the
pipeline cannot tell apart complete 79.1% against 50.9% of missions, though this experiment
cannot say how much of that belongs to the mechanics and how much to a delivery policy that had
scarcely seen a shallow grasp. Three quantities available at the moment the jaws are told to
shut decide whether the grasp will end engaged. One is a tolerance the part drawings fix before
any grasp is measured, and the grasps then confirm it: 90.5% against 42.5%. Nothing applied at
or after the closure raised completion — in both intervention arms the failures only moved to
earlier phases — though Section 6 records that neither arm could sharply have shown the
opposite.

The general lesson is that a pipeline guarded by boolean acceptance tests can be losing missions
to a continuous quantity that none of those tests reads. Four steps find it: measure a candidate
where control passes on; see whether the phases that follow leave it alone; sort their outcomes
by it; and work back to what fixed its value.

---

## Acknowledgement

(per template, not numbered)

---

## References

[1] R. Bähnemann, M. Pantic, M. Popović, D. Schindler, M. Tranzatto, M. Kamel, M. Grimm,
J. Widauer, R. Siegwart, and J. Nieto, "The ETH-MAV Team in the MBZ International Robotics
Challenge," J. Field Robot., Vol. 36, No. 1, pp. 78-103, January 2019. DOI: 10.1002/rob.21824.
[2] T. Baca, R. Penicka, P. Stepan, M. Petrlik, V. Spurny, D. Hert, and M. Saska, "Autonomous
Cooperative Wall Building by a Team of Unmanned Aerial Vehicles in the MBZIRC 2020 Competition,"
Robot. Auton. Syst., Vol. 167, 104482, September 2023. DOI: 10.1016/j.robot.2023.104482.
[3] K. Hang, X. Lyu, H. Song, J. A. Stork, A. M. Dollar, D. Kragic, and F. Zhang, "Perching and
Resting - A Paradigm for UAV Maneuvering with Modularized Landing Gears," Sci. Robot., Vol. 4,
No. 28, eaau6637, March 2019. DOI: 10.1126/scirobotics.aau6637.
[4] S. Ubellacker, A. Ray, J. M. Bern, J. Strader, and L. Carlone, "High-Speed Aerial Grasping
Using a Soft Drone with Onboard Perception," npj Robot., Vol. 2, No. 1, 5, August 2024. DOI:
10.1038/s44182-024-00012-1.
[5] F. Bottarel, G. Vezzani, U. Pattacini, and L. Natale, "GRASPA 1.0: GRASPA is a Robot Arm
graSping Performance BenchmArk," IEEE Robot. Autom. Lett., Vol. 5, No. 2, pp. 836-843, April
2020. DOI: 10.1109/LRA.2020.2965865.
[6] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. Aparicio Ojea, and
K. Goldberg, "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and
Analytic Grasp Metrics," in Proc. Robotics: Science and Systems (RSS), July 2017. DOI:
10.15607/RSS.2017.XIII.058.
[7] R. R. Burridge, A. A. Rizzi, and D. E. Koditschek, "Sequential Composition of Dynamically
Dexterous Robot Behaviors," Int. J. Robot. Res., Vol. 18, No. 6, pp. 534-555, June 1999. DOI:
10.1177/02783649922066385.
[8] Y. Lee, J. J. Lim, A. Anandkumar, and Y. Zhu, "Adversarial Skill Chaining for Long-Horizon
Robot Manipulation via Terminal State Regularization," in Proc. 5th Conference on Robot Learning
(CoRL), PMLR Vol. 164, pp. 406-416, November 2021.
[9] G. Eoh, "Deep-Reinforcement-Learning-Based Object Transportation Using Task Space
Decomposition," Sensors, Vol. 23, No. 10, 4807, May 2023. DOI: 10.3390/s23104807.
[10] S. Ross, G. Gordon, and D. Bagnell, "A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning," in Proc. 14th International Conference on Artificial
Intelligence and Statistics (AISTATS), PMLR Vol. 15, pp. 627-635, April 2011.
[11] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy
Optimization Algorithms," arXiv:1707.06347, July 2017.
[12] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo,
H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg, "Orbit: A Unified
Simulation Framework for Interactive Robot Learning Environments," IEEE Robot. Autom. Lett.,
Vol. 8, No. 6, pp. 3740-3747, June 2023. DOI: 10.1109/LRA.2023.3270034.
[13] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, "Sim-to-Real Transfer of Robotic
Control with Dynamics Randomization," in Proc. IEEE International Conference on Robotics and
Automation (ICRA), pp. 3803-3810, May 2018. DOI: 10.1109/ICRA.2018.8460528.
[14] E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference,"
J. Amer. Statist. Assoc., Vol. 22, No. 158, pp. 209-212, June 1927. DOI:
10.1080/01621459.1927.10502953.
[15] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel,
P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M.
Perrot, and E. Duchesnay, "Scikit-learn: Machine Learning in Python," J. Mach. Learn. Res., Vol.
12, pp. 2825-2830, 2011.
