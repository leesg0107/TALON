# TALON — Autonomous Aerial Grasping with Dual-Purpose Landing Gear

A task-decomposed control pipeline for autonomous drone grasping and transport in Isaac Lab: **RL waypoint navigation + analytical PID docking + RL loaded delivery**.

https://github.com/leesg17/soltronev3/raw/main/assets/end-to-end.webm

## Results

End-to-end evaluation over **500 missions** in 128 parallel environments:

| Phase | Description | Success Rate |
|-------|-------------|-------------|
| Approach → Dock | RL flight model follows waypoints to above box | **500/500 (100%)** |
| Dock → Climb | Analytical PID controller grasps box and climbs | **493/500 (98.6%)** |
| Climb → Delivery | Payload retained during ascent to 1.0m | **446/493 (90.5%)** |
| Delivery → Done | RL loaded model delivers box to target | **361/446 (80.9%)** |
| **Full End-to-End** | Complete mission success | **361/500 (72.2%)** |

## Pipeline Architecture

```
Approach (RL)  →  Analytical Dock + Climb  →  Delivery (RL)
 22D obs           PID Controller              23D obs
 4D action         (gain-scheduled)            4D action
 flight model                                  loaded model
```

1. **Approach**: RL flight policy follows 3–4 waypoints to 50 cm above the box
2. **Dock**: Analytical PID controller descends with XY-gated descent, closes gripper around box
3. **Climb**: PID controller (low-gain) lifts drone+box to 1.0 m safe altitude
4. **Delivery**: RL loaded-flight policy follows 3–4 waypoints to deliver box to target

The drone uses its own **landing gear as a parallel gripper** (dual-purpose design). The gripper opening is 10 cm with an 8 cm target object, leaving only **1 cm clearance per side** — too tight for RL to solve reliably, which motivates the analytical docking controller.

---

## Setup

### Prerequisites

- **Isaac Sim 4.5.1** + **Isaac Lab** ([installation guide](https://isaac-sim.github.io/IsaacLab/))
- Python 3.10+ (conda recommended)
- PyTorch 2.x with CUDA
- [SKRL](https://skrl.readthedocs.io/) RL library

### Installation

```bash
git clone https://github.com/leesg17/soltronev3.git
cd soltronev3
pip install skrl
```

---

## Training Guide

### Step 1: Train Approach Model (Flight)

Train a drone to follow waypoints without payload. Trained from scratch.

```bash
python train_gripper_waypoint.py --mode flight --num_envs 4096 \
  --log_dir models/my_flight
```

- **Duration**: ~1B steps (~2 hours on RTX 4090)
- **Observation**: 22D (body velocity, angular velocity, rotation matrix, goal in body frame, prev action)
- **Action**: 4D (acceleration xyz + yaw reference)
- **Expected**: `best_agent.pt` at 70–80% of training, 20+ goals/episode

**Verify:**
```bash
python scripts/eval_gripper_wp.py --num-envs 64 --episodes 200
# Expected: ~20 goals/episode, 0% crash
```

### Step 2: Train Loaded Flight (Base)

Train loaded flight with box ideally placed in gripper. Warm-start from flight model.

```bash
python train_gripper_waypoint.py --mode loaded --num_envs 4096 \
  --warm_start models/my_flight/best_agent.pt \
  --reset_std \
  --log_dir models/my_loaded_base
```

- `--warm_start`: Copies flight model weights (22D→23D, 23rd input initialized to zero)
- `--reset_std`: Resets exploration noise to 1.0 (required for new task adaptation)
- **Duration**: ~1B steps
- Box is placed **ideally centered** in gripper during this phase

**Verify:**
```bash
python scripts/eval_gripper_wp.py --loaded --num-envs 64 --episodes 200
# Expected: ~19 goals/episode
```

### Step 3: Generate Physical Grasp States

Run the analytical docking controller to collect ~5,000 physically-realistic grasped states.

```bash
python scripts/generate_grasp_states.py
# → data/grasp_states.pt (~5 min, 128 parallel envs)
# ~97% dock success rate, diverse grasp configurations
```

This captures real physical states after dock+climb:
- Drone position/orientation/velocity at z ≈ 1.2 m
- Box position/orientation (as physically held by gripper)
- Joint angles (gripper closed around box)

### Step 4: Fine-tune with Physical Grasp States

Fine-tune the loaded model on realistic grasp states. Loads `data/grasp_states.pt` automatically.

```bash
python train_gripper_waypoint.py --mode loaded --num_envs 4096 \
  --checkpoint models/my_loaded_base/best_agent.pt \
  --log_dir models/my_loaded_grasp
```

- No `--reset_std` needed (same task, only initial state distribution changes)
- **Duration**: 500M–1B steps
- Model adapts to asymmetric grasps, tilted box, varied payload dynamics

### Step 5 (Optional): Overshoot Prevention Fine-tuning

Further fine-tuning to reduce aggressive maneuvers during delivery.

```bash
python train_gripper_waypoint.py --mode loaded --num_envs 4096 \
  --checkpoint models/my_loaded_grasp/best_agent.pt \
  --max_steps 500_000_000 \
  --log_dir models/my_loaded_final
```

---

## Evaluation

### End-to-End (Parallel, Headless)

```bash
python scripts/eval_mission_headless.py
# 128 environments, 500 missions
# Outputs: per-phase success rates, failure breakdown
```

### End-to-End (Single Env, Rendering)

```bash
python scripts/eval_mission.py
# Visual rendering of drone missions
```

### Standalone Model Evaluation

```bash
python scripts/eval_gripper_wp.py --num-envs 64 --episodes 200          # Flight
python scripts/eval_gripper_wp.py --loaded --num-envs 64 --episodes 200  # Loaded
```

### Analytical Dock Standalone

```bash
python scripts/eval_pd_dock.py
# Tests analytical dock+climb success rate in isolation
# 128 environments, 200 attempts
```

---

## Technical Documentation

For detailed system architecture, reward design, analytical controller, and domain randomization:

→ **[docs/technical_doc.md](docs/technical_doc.md)**

---

## Project Structure

```
soltronev3/
├── train_gripper_waypoint.py      # Training script (flight & loaded)
├── envs/
│   ├── drone_env.py               # GripperDroneEnv: physics + analytical dock
│   ├── gripper_waypoint_env.py    # GripperWaypointEnv: RL training wrapper
│   └── env_cfg.py                 # Environment configuration
├── agents/
│   └── waypoint_ppo_cfg.py        # PPO agent (2×128 MLP)
├── controllers/
│   └── drone_ctrl.py              # AttitudeController + MotorModel (300Hz)
├── rewards/
│   └── reward_fn.py               # Stage-specific reward functions
├── scripts/
│   ├── eval_mission.py            # End-to-end eval (rendering)
│   ├── eval_mission_headless.py   # End-to-end eval (parallel)
│   ├── eval_gripper_wp.py         # Standalone model eval
│   ├── eval_pd_dock.py            # Analytical dock eval
│   └── generate_grasp_states.py   # Physical grasp data generation
├── data/
│   └── grasp_states.pt            # ~5000 physical grasp states
├── assets/                        # URDF, meshes, demo video
└── docs/
    └── technical_doc.md           # Detailed technical documentation
```

## Key Specs

- **Physics**: Isaac Lab + PhysX 5, 300 Hz sim / 150 Hz policy
- **Drone**: 1.08 kg quad, SO(3) attitude PD inner loop
- **Box**: 8 cm cube, 0.2 kg, dynamic rigid body
- **Gripper clearance**: 1 cm per side (X-axis)
- **RL**: PPO (SKRL), 4,096 parallel envs, ~1B steps
- **Models**: 2×128 MLP, ELU, RunningStandardScaler

## Known Limitations

- **Delivery tilt**: Aggressive turns during loaded flight can exceed 70° → mission termination
- **Box drop**: Physical grasp quality varies — ~10% of deliveries lose the box
- **Climb timeout**: ~9% of climbs fail to reach 1.0 m altitude
- **Simulation only**: No sim-to-real transfer has been attempted

## License

MIT
