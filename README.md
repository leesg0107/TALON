# TALON — Autonomous Aerial Grasping with Dual-Purpose Landing Gear

A quadrotor that **uses its own landing gear as a gripper** to autonomously pick up, transport, and deliver objects — no dedicated manipulator needed.

The drone's two structural plates double as parallel gripper fingers: open for landing, closed for grasping. With only **1 cm of clearance per side** around an 8 cm target object, the system combines RL-trained navigation with an analytical docking controller to achieve end-to-end autonomous grasping and transport in simulation.

> **Demo**: Full autonomous mission — approach, dock, grasp, climb, and deliver.

<video src="https://github.com/leesg0107/TALON/raw/main/assets/end-to-end.mp4" controls width="100%"></video>

---

## Results

Each mission consists of the full cycle: approach the target → dock and grasp → climb with payload → deliver to a goal location. Evaluated over **500 missions** in 128 parallel Isaac Lab environments with domain randomization:

| Phase Transition | What Happens | Success Rate |
|-----------------|-------------|-------------|
| **Approach → Dock** | RL policy navigates through waypoints to 50 cm above the target | **500/500 (100%)** |
| **Dock → Climb** | Analytical PID controller descends, aligns, and closes gripper around the box | **493/500 (98.6%)** |
| **Climb → Delivery** | Drone ascends to 1.0 m with payload; box must stay in gripper | **446/493 (90.5%)** |
| **Delivery → Done** | RL loaded-flight policy navigates to delivery location with payload | **361/446 (80.9%)** |
| **Full End-to-End** | Complete autonomous mission from start to delivery | **361/500 (72.2%)** |

---

## How It Works

```
Approach (RL)  →  Dock + Climb (Analytical)  →  Delivery (RL)
 22D obs           PID with gain scheduling       23D obs (+payload mass)
 4D action         Gated descent + auto-grip      4D action
 flight model      0% crash rate                  loaded model
```

| Phase | Controller | What It Does |
|-------|-----------|-------------|
| **Approach** | RL (PPO) | Follows 3–4 waypoints to position above the target. Handles wind disturbance. |
| **Dock** | Analytical PID | Aligns XY over the box, descends through a two-stage gated descent, closes gripper when box is inside. |
| **Climb** | Analytical PID (low-gain) | Gently ascends to 1.0 m. Low gains prevent shaking the payload loose. |
| **Delivery** | RL (PPO) | Follows waypoints to the delivery target. Adapts to added payload mass (0.15–0.25 kg). |

**Why not RL for everything?** We tried — RL achieves only 24.8% docking success on dynamic objects despite extensive reward engineering. The 1 cm clearance requires geometric precision that RL exploration cannot reliably provide. See [docs/technical_doc.md §4](docs/technical_doc.md#4-rl-docking-attempt-stage-3) for the full failure analysis.

---

## Setup

### Prerequisites

- **Isaac Sim 4.5.1** + **Isaac Lab** ([installation guide](https://isaac-sim.github.io/IsaacLab/))
- Python 3.10+ (conda recommended)
- PyTorch 2.x with CUDA
- [SKRL](https://skrl.readthedocs.io/) RL library

### Installation

```bash
git clone https://github.com/leesg0107/TALON.git
cd TALON
pip install skrl
```

---

## Training Guide

Training proceeds in stages — each stage builds on the previous one.

### Step 1: Train Approach Model (Flight)

Train a drone to follow waypoints without payload. Trained from scratch.

```bash
python train_gripper_waypoint.py --mode flight --num_envs 4096 \
  --log_dir models/my_flight
```

- **Duration**: ~1B steps (~2 hours on RTX 4090)
- **Observation**: 22D (body velocity, angular velocity, rotation matrix, goal in body frame, prev action)
- **Action**: 4D (body-frame acceleration xyz + yaw angle reference)
- **Expected**: `best_agent.pt` at 70–80% of training, 20+ goals/episode

```bash
# Verify
python scripts/eval_gripper_wp.py --num-envs 64 --episodes 200
# Expected: ~20 goals/episode, 0% crash
```

### Step 2: Train Loaded Flight (Base)

Train loaded flight with box ideally placed in gripper. **Warm-start from the flight model** — training loaded flight from scratch fails because the drone must first know how to fly.

```bash
python train_gripper_waypoint.py --mode loaded --num_envs 4096 \
  --warm_start models/my_flight/best_agent.pt \
  --reset_std \
  --log_dir models/my_loaded_base
```

- `--warm_start`: Copies flight model weights (22D→23D, new payload input initialized to zero)
- `--reset_std`: Resets exploration noise to 1.0 (needed because the old policy is near-deterministic)
- **Duration**: ~1B steps
- Box is placed **ideally centered** in gripper during this phase

```bash
# Verify
python scripts/eval_gripper_wp.py --loaded --num-envs 64 --episodes 200
# Expected: ~19 goals/episode
```

### Step 3: Generate Physical Grasp States

Step 2 trains with ideal box placement. Real docking produces asymmetric, tilted grasps. This step collects ~5,000 realistic grasped states by running the analytical docking controller:

```bash
python scripts/generate_grasp_states.py
# → data/grasp_states.pt (~5 min, 128 parallel envs)
```

Each saved state includes drone pose/velocity, box pose (as physically held by gripper), and joint angles after a real dock+climb sequence.

### Step 4: Fine-tune with Physical Grasp States

Fine-tune the loaded model so it can handle realistic (non-ideal) grasps. `data/grasp_states.pt` is loaded automatically.

```bash
python train_gripper_waypoint.py --mode loaded --num_envs 4096 \
  --checkpoint models/my_loaded_base/best_agent.pt \
  --log_dir models/my_loaded_grasp
```

- **Duration**: 500M–1B steps
- Model adapts to asymmetric grasps, tilted box, varied payload dynamics

### Step 5 (Optional): Overshoot Prevention

Further fine-tuning to reduce aggressive waypoint approach maneuvers during delivery (which cause tilt → payload drop).

```bash
python train_gripper_waypoint.py --mode loaded --num_envs 4096 \
  --checkpoint models/my_loaded_grasp/best_agent.pt \
  --max_steps 500_000_000 \
  --log_dir models/my_loaded_final
```

---

## Evaluation

```bash
# End-to-end: 128 parallel environments, 500 missions
python scripts/eval_mission_headless.py

# End-to-end: single environment with rendering
python scripts/eval_mission.py

# Standalone RL model evaluation
python scripts/eval_gripper_wp.py --num-envs 64 --episodes 200          # Flight
python scripts/eval_gripper_wp.py --loaded --num-envs 64 --episodes 200  # Loaded

# Standalone analytical docking evaluation
python scripts/eval_pd_dock.py
```

---

## Technical Documentation

For detailed system architecture, reward functions, analytical controller design, and domain randomization:

→ **[docs/technical_doc.md](docs/technical_doc.md)**

---

## Project Structure

```
TALON/
├── train_gripper_waypoint.py        # RL training script (flight & loaded modes)
│
├── envs/
│   ├── drone_env.py                 # Core environment: physics, analytical
│   │                                #   docking controller, gripper logic
│   ├── gripper_waypoint_env.py      # RL training wrapper: 22D/23D obs,
│   │                                #   4D action, waypoint trajectory gen
│   └── env_cfg.py                   # Configuration: drone params, DR, scenes
│
├── controllers/
│   └── drone_ctrl.py                # Inner loop: SO(3) attitude controller,
│                                    #   motor model, mixer matrix (300 Hz)
├── agents/
│   └── waypoint_ppo_cfg.py          # PPO agent: 2×128 MLP, hyperparameters
│
├── rewards/
│   └── reward_fn.py                 # Reward functions for all stages
│
├── scripts/
│   ├── eval_mission_headless.py     # End-to-end eval (128 parallel envs)
│   ├── eval_mission.py              # End-to-end eval (single env, rendering)
│   ├── eval_gripper_wp.py           # Standalone RL model eval
│   ├── eval_pd_dock.py              # Standalone analytical dock eval
│   └── generate_grasp_states.py     # Generate physical grasp states for Stage 4
│
├── data/
│   └── grasp_states.pt              # ~5,000 pre-simulated grasp states
│
├── assets/
│   ├── gripper_drone_v5.urdf        # Quadrotor + dual-purpose landing gear
│   └── end-to-end.mp4              # Demo video
│
└── docs/
    └── technical_doc.md             # Full technical documentation
```

## Key Specs

| | |
|---|---|
| **Simulator** | Isaac Lab (Isaac Sim 4.5.1) + PhysX 5 |
| **Physics rate** | 300 Hz simulation, 150 Hz policy (decimation = 2) |
| **Drone** | 1.08 kg, X-config quadrotor, SO(3) attitude PD inner loop |
| **Target object** | 8 cm cube, 0.2 kg, dynamic rigid body |
| **Gripper clearance** | 1 cm per side (X-axis), 6.4 cm per side (Y-axis) |
| **RL algorithm** | PPO (SKRL), 4,096 parallel environments |
| **Network** | 2×128 MLP, ELU activation, orthogonal init |
| **Training** | ~1B steps per model (~2 hours on RTX 4090) |
| **Domain randomization** | Mass ±10%, motor thrust ±15%, wind 0.5 N, sensor noise |

## Known Limitations

- **Simulation only** — no sim-to-real transfer attempted
- **Single object type** — 8 cm cube only; no shape generalization
- **Delivery tilt** — aggressive turns during loaded flight can exceed 70° → box drop
- **Climb failure** — ~9% of climbs fail (box slips during ascent)

## License

MIT
