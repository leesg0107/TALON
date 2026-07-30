"""Build the IJASC paper-proposal PDF (text + embedded figures) with matplotlib."""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P11 = os.path.join(ROOT, "data/final_presentation/11_composition_findings")
P10 = os.path.join(ROOT, "data/final_presentation/10_prescription_validation")
PV2 = os.path.join(ROOT, "data/final_presentation/v2_plots")
OUT = os.path.join(ROOT, "TALON_paper_proposal.pdf")
A4 = (8.27, 11.69)
GREEN = "black"

# ---------------- content ----------------
TITLE = "Grasp Depth as a Hidden Interface Variable\nin a Modular Aerial Grasp-and-Transport System"
SUB = "Paper Proposal (IJASC)  ·  School of AI Software  ·  20211539 Solgyu Lee"

CONTRIB = [
 ("1. System.",
  "A complete simulated autonomous aerial grasp-and-transport system whose landing-gear struts double as "
  "a parallel gripper: RL waypoint navigation composed with analytical PID docking and climb (adopted "
  "because end-to-end RL docking plateaus at 24.8%), an 8-phase mission state machine, and ~66-72% end-to-end "
  "success across unseeded runs (72.2% on the shown 500-mission run) at a 10 mm strut clearance. The working "
  "system is both the "
  "applied result and the instrument for the analysis that follows."),
 ("2. Diagnosis of a hidden interface variable.",
  "I localize the end-to-end loss not to any stage but to a single continuous interface variable - grasp "
  "depth - and characterize it as (i) invisible to per-stage success metrics (0 of 383 handoffs rejected on "
  "depth; 37% shallow admitted), (ii) stochastically realized (pre-contact prediction R^2=0.066), (iii) "
  "irreversible downstream (climb->transport depth correlation 0.9994), (iv) class-specific and orthogonal "
  "(depth->transport p=1e-12; tilt->climb p=7e-12), and (v) dominant (deep-grasp success 79.1% vs shallow "
  "53.3%). The same variable resolves the standalone-vs-pipeline paradox as a covariate shift (training bank "
  "99.2% deep vs live pipeline 37% shallow; deep-conditional 91.1% ~ standalone ~88%)."),
 ("3. No-retrain intervention analysis.",
  "Under a modularity premise (compose pretrained skills without retraining), I evaluate representative "
  "implementations of the actions available on a realized interface variable - reshape (control shaping), "
  "reject (handoff gate), and resample (re-grasp) - and show on same-seed mission layouts that none raises "
  "end-to-end success: reshaping backfires (deep fraction 47.1%->0.0%), gating certifies-but-rejects "
  "(converting failures into rejections, not successes), and re-grasp leaves end-to-end success statistically "
  "unchanged (63.0% vs off 69.0%, Fisher p=0.16, n.s.) while relocating failure from transport (18.0%->10.4%, "
  "p=0.02) to dock (0.3%->16.1%). The common pattern is risk relocation, not risk removal. Re-grasp additionally "
  "serves as an interventional "
  "confirmation of the diagnosis: re-drawing depth reduces exactly the depth-matched failure class."),
 ("4. Interface monitoring as a design tool.",
  "The same handoff signals that fail as fixes succeed as monitors: entry-state signals predict their matched "
  "failure class (AUC 0.76-0.84), and conditional acceptance trades throughput for certified reliability "
  "(success on accepted handoffs rises 69.2%->75.2%->88.5% as acceptance tightens). The negative results thus "
  "define a throughput-vs-certified-reliability operating frontier rather than a dead end."),
 ("5. Design guidelines (from one worked case).",
  "Instrument continuous interface quality alongside boolean stage success; test irreversibility before "
  "investing in downstream repair; treat interface quality as a training-distribution variable; budget for "
  "reject/retry rather than control-shaping; and measure, do not model, interventions (an analytic ~78% "
  "re-grasp projection was falsified by a measured 63%)."),
]

DIFF = [
 ("1. Funnel / sequential composition (Burridge-Rizzi-Koditschek 1999; LQR-Trees 2010)",
  "certify composition by boolean set-containment at boundaries - my data falsifies the sufficiency of that "
  "predicate: a continuous quality variable inside the accepted set governs success."),
 ("2. Skill-chaining & skill-based meta-RL (Konidaris & Barto 2009; recent 2025)",
  "repair handoffs by reshaping the boundary / skill distribution during training (e.g. Self-Improving Skill "
  "Learning 2025) - I instead show the dominant coupling is a stochastic, irreversible latent scalar the "
  "downstream policy silently accepts, and quantify what each no-retrain fix actually does; my survivor-filtered "
  "training bank is itself an instance of the distribution mismatch those methods aim to fix."),
 ("3. RL aerial grasping (Swooper 2026; Flying Hand 2025; SNAG 2021)",
  "optimize an isolated grasp - a single end-to-end policy reporting binary, isolated grasp success and "
  "sim-to-real transfer (e.g. Swooper 2026; Ubellacker & Carlone npj Robotics 2024). I do not compete on the "
  "grasp; I treat it as one "
  "stage of a multi-stage mission and quantify how its continuous quality propagates and caps downstream transport "
  "- and adopt a hybrid RL+analytical stack precisely because end-to-end RL docking plateaus (24.8%) at my 10 mm "
  "strut clearance."),
 ("4. Task-oriented grasp quality (Dex-Net 2.0; TOG-Net)",
  "selects grasps via a pre-contact quality model - in my system quality is stochastically realized at the "
  "contact event (R^2=0.066), so the value lies in post-hoc measurement and gating, not pre-selection."),
 ("5. Modular failure attribution (DAgger compounding error; REFLECT 2023)",
  "attributes loss to per-module error/shift - I localize loss to a single continuous variable that passes "
  "every module's success check and go beyond attribution to a controlled falsification of the standard fixes."),
]

STRUCTURE = [
  "Section 1:  Introduction",
  "Section 2:  Related Work",
  "Section 3:  Aerial Grasp-and-Transport System",
  "Section 4:  Diagnosing the Hidden Interface Variable",
  "Section 5:  No-Retrain Interventions at the Interface",
  "Section 6:  Discussion - Design Guidelines and Limitations",
  "Section 7:  Conclusion",
]

FIGS = [
  dict(n="Figure 1", title="The composition paradox",
       imgs=[f"{P11}/figA_stage_cascade.png", f"{P11}/figC_success_vs_depth.png"], layout=(2, 1),
       cap="Every stage succeeds >=80% yet the pipeline caps near 72% (72.2% in the shown 500-mission run, top); "
           "end-to-end success is governed by the inherited grasp depth - deep 79.1% vs shallow 53.3% "
           "(bottom; N=383 diagnosis cohort)."),
  dict(n="Figure 2", title="Characterizing the hidden interface variable",
       imgs=[f"{P11}/figE_irreversibility_scatter.png", f"{P11}/figF_class_specificity_2x2.png",
             f"{P11}/figB_covariate_shift.png", f"{P11}/figG_depth_lockin.png"], layout=(2, 2),
       cap="(a) Irreversibility: climb-entry vs transport-entry depth, r=0.9994.  (b) Class-specificity: "
           "depth->transport, tilt->climb, orthogonal.  (c) Covariate shift: 99.2%-deep training bank vs "
           "37%-shallow live distribution.  (d) Depth lock-in during the grasp."),
  dict(n="Figure 3", title="No-retrain interventions relocate risk, not remove it  (+ Table 1)",
       imgs=[f"{P11}/figI_intervention_taxonomy.png"], layout=(1, 1),
       cap="Reshape backfires, gate certifies-but-rejects, re-grasp moves failure from transport to dock; none "
           "raises end-to-end success. (The full intervention ledger with 95% CIs and tests is Table 1.)"),
  dict(n="Figure 4", title="Interface monitoring ROC",
       imgs=[f"{P10}/fig2_gate_roc.png"], layout=(1, 1),
       cap="Entry-state signals predict their matched failure class (AUC 0.76-0.84); the signals that fail as "
           "fixes succeed as monitors on a throughput-vs-certified-reliability frontier."),
]

# ---------------- layout helpers ----------------
def new_page():
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    return fig

def wrapped(fig, y, lead, body, x0=0.07, w=104, lh=0.0150, size=9.2):
    fig.text(x0, y, lead, fontsize=size + 0.6, weight="bold", va="top", ha="left", color="black")
    y -= lh * 1.15
    for line in textwrap.wrap(body, w):
        fig.text(x0 + 0.015, y, line, fontsize=size, va="top", ha="left", color="black")
        y -= lh
    return y - lh * 0.7

def heading(fig, y, text, x0=0.07, lh=0.0150):
    y -= lh * 0.4
    fig.text(x0, y, text, fontsize=12.5, weight="bold", va="top", ha="left", color=GREEN)
    return y - lh * 1.7

# ---------------- build ----------------
with PdfPages(OUT) as pdf:
    # Page 1 — title + contributions
    fig = new_page()
    fig.text(0.07, 0.965, TITLE, fontsize=15.5, weight="bold", va="top", ha="left", color="black")
    fig.text(0.07, 0.905, SUB, fontsize=9.5, style="italic", va="top", ha="left", color="black")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.888, 0.888], color=GREEN, lw=1.4, transform=fig.transFigure))
    y = 0.86
    y = heading(fig, y, "Main Contributions")
    for lead, body in CONTRIB:
        y = wrapped(fig, y, lead, body)
    pdf.savefig(fig); plt.close(fig)

    # Page 2 — differences + structure
    fig = new_page()
    y = 0.955
    y = heading(fig, y, "Main Differences from Prior Research")
    for lead, body in DIFF:
        y = wrapped(fig, y, lead, body)
    y -= 0.015
    y = heading(fig, y, "Paper Structure")
    for s in STRUCTURE:
        fig.text(0.085, y, s, fontsize=10.5, va="top", ha="left", color="black")
        y -= 0.0175
    fig.text(0.07, 0.045,
             "Scope notes: the system transports the payload to a target location (no release/placement is "
             "modeled). 'No-retrain fix' (not 'fundamental ceiling'); representative (not exhaustive) "
             "implementations; single-system, mostly single-seed case study. End-to-end success is unseeded "
             "(66-72% run-to-run); the Section 5 interventions use a separate seed-1 baseline (69.0%, N=384), so "
             "absolute rates are not paired across sections (diagnosis N=383, interventions N=384/192, system N=500).",
             fontsize=7.6, style="italic", va="bottom", ha="left", color="black", wrap=True)
    pdf.savefig(fig); plt.close(fig)

    # Figure pages
    for F in FIGS:
        fig = new_page()
        fig.text(0.07, 0.965, f'{F["n"]}   —   {F["title"]}', fontsize=12.5, weight="bold",
                 va="top", ha="left", color="black")
        rows, cols = F["layout"]
        gs = fig.add_gridspec(rows, cols, left=0.06, right=0.94, top=0.90, bottom=0.20,
                              hspace=0.12, wspace=0.08)
        for i, ip in enumerate(F["imgs"]):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            try:
                ax.imshow(mpimg.imread(ip))
            except Exception as e:
                ax.text(0.5, 0.5, f"[missing: {os.path.basename(ip)}]", ha="center")
            ax.axis("off")
        cap = "\n".join(textwrap.wrap(F["cap"], 118))
        fig.text(0.07, 0.155, cap, fontsize=8.6, va="top", ha="left", color="black")
        pdf.savefig(fig, dpi=200); plt.close(fig)

    d = pdf.infodict()
    d["Title"] = "TALON — Grasp Depth as a Hidden Interface Variable (IJASC Proposal)"
    d["Author"] = "Solgyu Lee"

print("wrote", OUT)
