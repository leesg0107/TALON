"""Stage 3 → 4 transition demo: dock → grasp → loaded flight.

Flow per env:
  Phase 1 (Stage 3 model): approach box, dock, gripper closes
  Phase 2 (Stage 4 model): fly to delivery goal with grasped box

Environment runs as Stage 3 (dynamic box, pedestal) throughout.
Model switch happens per-env when docking is confirmed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ---- Config: Stage 3 env with dynamic box ----
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 4
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 60.0  # long episode for full sequence
cfg.lock_gripper = True       # auto-close on dock (same as training)

# Box stays kinematic (same as Stage 3 training).
# After dock, we manually teleport box to gripper each step for flight phase.

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device
num_envs = env.num_envs

# ---- Load both models ----
STAGE3_CKPT = "logs/best_models/best_fc34_loiter_tilt.pt"
STAGE4_CKPT = "logs/stage4_pre_grasped_v9/final_agent.pt"

agent_s3 = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                           checkpoint_path=STAGE3_CKPT)
agent_s3.set_running_mode("eval")

agent_s4 = build_ppo_agent(env=env_wrapped, device=device, stage=4,
                           checkpoint_path=STAGE4_CKPT)
agent_s4.set_running_mode("eval")

# ---- Per-env state machine ----
# States: "approach" → "dock_wait" → "flight" → "arrived"
state = ["approach"] * num_envs
align_count = torch.zeros(num_envs, dtype=torch.long, device=device)
gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)

# Stage 4 delivery goals (random, set when transitioning)
delivery_goal = torch.zeros(num_envs, 3, device=device)

# Results
results = {
    "dock_success": 0,
    "flight_success": 0,
    "total_episodes": 0,
    "dock_times": [],
    "flight_times": [],
}

# ---- Helpers ----
DOCK_THRESHOLD = 300       # cumulative overlap steps for dock (~2s)
GRIP_SETTLE_STEPS = 150    # 1s wait after dock for gripper to close firmly
ARRIVAL_DIST = 0.3         # m, close enough to delivery goal

def compute_overlap(env, device, num_envs):
    """Compute overlap ratio in gripper local frame."""
    pos_w = env.robot.data.root_pos_w
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=device)
    gripper_pos_w = pos_w + torch.bmm(R, gripper_offset.expand(num_envs, -1).unsqueeze(-1)).squeeze(-1)

    box_local = torch.bmm(R.transpose(1, 2), (env.object_pos - gripper_pos_w).unsqueeze(-1)).squeeze(-1)

    box_half = 0.04
    gx, gy = 0.05, 0.062

    ox = (torch.min(torch.full_like(box_local[:, 0], gx), box_local[:, 0] + box_half)
        - torch.max(torch.full_like(box_local[:, 0], -gx), box_local[:, 0] - box_half)).clamp(min=0)
    oy = (torch.min(torch.full_like(box_local[:, 1], gy), box_local[:, 1] + box_half)
        - torch.max(torch.full_like(box_local[:, 1], -gy), box_local[:, 1] - box_half)).clamp(min=0)

    z_ok = ((box_local[:, 2] > -0.12) & (box_local[:, 2] < 0.02)).float()
    return (ox * oy) / (2 * box_half) ** 2 * z_ok, gripper_pos_w

def sample_delivery_goal(env_id):
    """Sample a random delivery goal for Stage 4 flight."""
    xy = (torch.rand(2, device=device) * 2.0 - 1.0) * 1.0  # ±1m
    z = torch.tensor(1.5, device=device)
    return torch.cat([xy, z.unsqueeze(0)])

# ---- Main loop ----
max_steps = 9000  # 60s at 150Hz
obs, _ = env_wrapped.reset()
step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
dock_step = torch.zeros(num_envs, dtype=torch.long, device=device)  # step when docked

print(f"\n{'='*60}")
print(f"  Stage 3 → 4 Transition Demo")
print(f"  Stage 3: {STAGE3_CKPT}")
print(f"  Stage 4: {STAGE4_CKPT}")
print(f"  Envs: {num_envs}, Max steps: {max_steps}")
print(f"{'='*60}\n")

for step in range(max_steps):
    overlap, gripper_pos_w = compute_overlap(env, device, num_envs)

    # ---- Get actions from appropriate model per-env ----
    with torch.no_grad():
        action_s3 = agent_s3.act(obs, timestep=step, timesteps=max_steps)[0]
        action_s4 = agent_s4.act(obs, timestep=step, timesteps=max_steps)[0]

    action = action_s3.clone()  # default: Stage 3

    for i in range(num_envs):
        ov = overlap[i].item()

        if state[i] == "approach":
            # Stage 3 model controls, gripper open
            action[i, 7] = 1.0  # open

            # Cumulative overlap check
            if ov > 0.50:
                align_count[i] += 1

            if align_count[i] >= DOCK_THRESHOLD:
                state[i] = "dock_wait"
                dock_step[i] = step_count[i].clone()
                gripper_closed[i] = True
                dt = step_count[i].item() / 150.0
                results["dock_success"] += 1
                results["dock_times"].append(dt)
                print(f"  [env{i}] step {step}: DOCKED ({dt:.1f}s) → closing gripper")

        elif state[i] == "dock_wait":
            # Wait for gripper to settle, keep Stage 3 model
            action[i, 7] = -1.0  # closed

            # CRITICAL: prevent contain_success truncation (450 steps)
            # Cap at 400 so it never reaches 450 while we're waiting
            if env.contain_hold_count[i] > 400:
                env.contain_hold_count[i] = 400

            wait_steps = step_count[i].item() - dock_step[i].item()

            if wait_steps >= GRIP_SETTLE_STEPS:
                # Switch to Stage 4 — set delivery goal
                delivery_goal[i] = sample_delivery_goal(i)
                env.goal_pos[i] = delivery_goal[i]

                # Update controller mass for payload
                drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[i]
                env.attitude_ctrl.mass[i] = drone_mass + 0.2  # box mass

                state[i] = "flight"
                print(f"  [env{i}] step {step}: GRIP SETTLED → Stage 4 flight to "
                      f"({delivery_goal[i, 0]:.2f}, {delivery_goal[i, 1]:.2f}, {delivery_goal[i, 2]:.1f})")

        elif state[i] == "flight":
            # Stage 4 model controls
            action[i] = action_s4[i]
            action[i, 7] = -1.0  # keep gripper closed

            # Prevent truncation and box_fell termination during flight
            if env.contain_hold_count[i] > 400:
                env.contain_hold_count[i] = 400
            env.object_pos[i, 2] = max(env.object_pos[i, 2].item(), 0.50)  # prevent box_fell

            # Check arrival
            dist = torch.norm(gripper_pos_w[i] - delivery_goal[i]).item()
            if dist < ARRIVAL_DIST:
                state[i] = "arrived"
                flight_time = (step_count[i].item() - dock_step[i].item() - GRIP_SETTLE_STEPS) / 150.0
                results["flight_success"] += 1
                results["flight_times"].append(flight_time)
                print(f"  [env{i}] step {step}: *** ARRIVED *** (flight {flight_time:.1f}s, dist={dist:.2f}m)")

        elif state[i] == "arrived":
            # Stay at goal, Stage 4 model
            action[i] = action_s4[i]
            action[i, 7] = -1.0
            # Prevent truncation
            if env.contain_hold_count[i] > 400:
                env.contain_hold_count[i] = 400

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    step_count += 1

    # Teleport box to gripper for flight/arrived envs (kinematic box doesn't follow gripper)
    for i in range(num_envs):
        if state[i] in ("flight", "arrived"):
            pos_w = env.robot.data.root_pos_w[i]
            quat_w = env.robot.data.root_quat_w[i]
            R_i = quat_to_rot_matrix(quat_w.unsqueeze(0))
            g_off = torch.tensor([0.0, 0.0, -0.08], device=device)
            g_pos = pos_w + (R_i[0] @ g_off)
            # Place box at gripper center (slightly below)
            box_state = torch.zeros(1, 13, device=device)
            box_state[0, :3] = g_pos
            box_state[0, 2] -= 0.04  # box center = gripper center - box_half
            box_state[0, 3] = 1.0    # quaternion w
            env_id = torch.tensor([i], device=device)
            env.grasp_object.write_root_state_to_sim(box_state, env_id)

    # ---- Status print ----
    if step % 450 == 0:  # every 3s
        status_parts = []
        for i in range(num_envs):
            ov = overlap[i].item()
            box_z = env.object_pos[i, 2].item()
            s = state[i][0].upper()  # A/D/F/R
            if state[i] == "flight":
                dist = torch.norm(gripper_pos_w[i] - delivery_goal[i]).item()
                status_parts.append(f"env{i}:{s} d={dist:.2f}")
            else:
                status_parts.append(f"env{i}:{s} ov={ov*100:.0f}% bz={box_z:.2f}")
        print(f"  [{step/150:.0f}s] {' | '.join(status_parts)}")

    # ---- Episode reset ----
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        results["total_episodes"] += len(reset_list)
        for i in reset_list:
            final_state = state[i]
            print(f"  [env{i}] step {step}: RESET (was {final_state})")
            state[i] = "approach"
            align_count[i] = 0
            gripper_closed[i] = False
            dock_step[i] = 0
            step_count[i] = 0
            # Reset controller mass
            env.attitude_ctrl.mass[i] = env.attitude_ctrl.base_mass * env.mass_scale[i]

# ---- Final Results ----
print(f"\n{'='*60}")
print(f"  STAGE 3 → 4 TRANSITION RESULTS")
print(f"{'='*60}")
print(f"  Total episodes:   {results['total_episodes'] + num_envs}")
print(f"  Dock success:     {results['dock_success']}")
if results["dock_times"]:
    import numpy as np
    print(f"  Avg dock time:    {np.mean(results['dock_times']):.1f}s")
print(f"  Flight success:   {results['flight_success']}")
if results["flight_times"]:
    print(f"  Avg flight time:  {np.mean(results['flight_times']):.1f}s")
full_success = results["flight_success"]
total = results["total_episodes"] + num_envs
print(f"  Full pipeline:    {full_success}/{total} ({100*full_success/max(total,1):.1f}%)")
print(f"{'='*60}")

env.close()
simulation_app.close()
