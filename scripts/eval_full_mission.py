"""Full mission evaluation: Approach → Dock → Grasp → Fly to target.
Uses Stage 3 model for approach+dock, Stage 4 model for loaded flight."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import math
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Use Stage 3 env (pedestal + dynamic box for real grasping)
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 4
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 60.0
cfg.lock_gripper = False
# Dynamic box: can be physically grasped and carried
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device
num_envs = env.num_envs

# Load BOTH models
agent_dock = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                             checkpoint_path="logs/best_models/best_fc34_loiter_tilt.pt")
agent_dock.set_running_mode("eval")

agent_fly = build_ppo_agent(env=env_wrapped, device=device, stage=4,
                            checkpoint_path="logs/best_models/best_stage4_loaded.pt")
agent_fly.set_running_mode("eval")

# State machine per env
# approach → dock → grip → fly → arrived
state = ["approach"] * num_envs
align_count = torch.zeros(num_envs, dtype=torch.long, device=device)
grip_count = torch.zeros(num_envs, dtype=torch.long, device=device)

# Delivery targets (random, set after dock)
delivery_target = torch.zeros(num_envs, 3, device=device)
delivery_set = torch.zeros(num_envs, dtype=torch.bool, device=device)

# Payload simulation (activated after dock)
payload_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
payload_mass = torch.full((num_envs,), 0.2, device=device)  # box mass
grasp_offset = torch.zeros(num_envs, 3, device=device)

# Results
from collections import Counter
episode_results = Counter()

max_steps = 9000  # 60s
print(f"\n=== Full Mission Evaluation: {num_envs} envs ===")
print(f"Stage 3 model: best_fc34_loiter_tilt.pt (dock)")
print(f"Stage 4 model: best_stage4_loaded.pt (fly)")
print()

obs, _ = env_wrapped.reset()

for step in range(max_steps):
    # Get actions from BOTH models
    with torch.no_grad():
        action_dock = agent_dock.act(obs, timestep=step, timesteps=max_steps)[0]
        action_fly = agent_fly.act(obs, timestep=step, timesteps=max_steps)[0]

    # Select action per env based on state
    action = action_dock.clone()
    for i in range(num_envs):
        if state[i] in ("fly", "arrived"):
            action[i] = action_fly[i]

    # Compute overlap for dock detection
    pos_w = env.robot.data.root_pos_w
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=device)
    gripper_pos_w = pos_w + torch.bmm(R, gripper_offset_b.expand(num_envs, -1).unsqueeze(-1)).squeeze(-1)

    box_offset_w = env.object_pos - gripper_pos_w
    box_local = torch.bmm(R.transpose(1, 2), box_offset_w.unsqueeze(-1)).squeeze(-1)

    box_half = 0.04
    gripper_half_x, gripper_half_y = 0.05, 0.062
    overlap_x = (torch.min(torch.full_like(box_local[:, 0], gripper_half_x), box_local[:, 0] + box_half)
               - torch.max(torch.full_like(box_local[:, 0], -gripper_half_x), box_local[:, 0] - box_half)).clamp(min=0.0)
    overlap_y = (torch.min(torch.full_like(box_local[:, 1], gripper_half_y), box_local[:, 1] + box_half)
               - torch.max(torch.full_like(box_local[:, 1], -gripper_half_y), box_local[:, 1] - box_half)).clamp(min=0.0)
    box_area = (2 * box_half) ** 2
    overlap_xy = (overlap_x * overlap_y) / box_area
    z_in_range = ((box_local[:, 2] > -0.12) & (box_local[:, 2] < 0.02)).float()
    overlap_ratio = overlap_xy * z_in_range

    for i in range(num_envs):
        ov = overlap_ratio[i].item()
        box_z = env.object_pos[i, 2].item()
        drone_z = pos_w[i, 2].item()

        if state[i] == "approach":
            action[i, 7] = 1.0  # gripper open
            if ov > 0.50:
                align_count[i] += 1
            if align_count[i] > 300:  # cumulative ~2s
                state[i] = "dock"
                grip_count[i] = 0
                print(f"  env{i} step {step}: DOCKED → closing gripper")

        elif state[i] == "dock":
            action[i, 7] = -1.0  # close gripper
            grip_count[i] += 1
            if grip_count[i] > 150:  # 1s to close
                state[i] = "fly"
                # Activate payload physics
                payload_active[i] = True
                grasp_offset[i] = box_local[i].clone()  # record grasp offset in gripper frame
                # Set payload mass on drone (for observation)
                env.payload_mass[i] = 0.2
                env.is_grasped[i] = True
                # Set delivery target
                delivery_target[i, 0] = gripper_pos_w[i, 0].item() + (torch.rand(1).item() - 0.5) * 2.0
                delivery_target[i, 1] = gripper_pos_w[i, 1].item() + (torch.rand(1).item() - 0.5) * 2.0
                delivery_target[i, 2] = 1.5
                delivery_set[i] = True
                env.goal_pos[i] = delivery_target[i]
                print(f"  env{i} step {step}: GRIPPED → flying to [{delivery_target[i, 0]:.1f}, {delivery_target[i, 1]:.1f}, {delivery_target[i, 2]:.1f}]")

        elif state[i] == "fly":
            action[i, 7] = -1.0  # keep gripper closed
            # Check arrival
            dist_to_target = torch.norm(gripper_pos_w[i] - delivery_target[i]).item()
            if dist_to_target < 0.5:
                state[i] = "arrived"
                print(f"  env{i} step {step}: *** ARRIVED at target! *** dist={dist_to_target:.2f}m")

        elif state[i] == "arrived":
            action[i, 7] = -1.0  # keep holding

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    # Box is dynamic — physics handles everything (gripper contact holds box)

    # Status
    if step % 750 == 0:
        status = ""
        for i in range(num_envs):
            box_z = env.object_pos[i, 2].item()
            if state[i] == "fly" and delivery_set[i]:
                dist = torch.norm(gripper_pos_w[i] - delivery_target[i]).item()
                status += f"  env{i}:{state[i]}(dist={dist:.2f})"
            else:
                status += f"  env{i}:{state[i]}(bz={box_z:.2f})"
        print(f"  [{step/150:.0f}s]{status}")

    # Episode reset
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        for i in reset_list:
            episode_results[state[i]] += 1
            state[i] = "approach"
            align_count[i] = 0
            grip_count[i] = 0
            delivery_set[i] = False
        obs, _ = env_wrapped.reset()

# Final
for i in range(num_envs):
    episode_results[state[i]] += 1

total = sum(episode_results.values())
print(f"\n{'='*60}")
print(f"  FULL MISSION RESULTS")
print(f"{'='*60}")
print(f"  Total episodes: {total}")
for s in ["arrived", "fly", "dock", "approach"]:
    c = episode_results.get(s, 0)
    if c > 0:
        print(f"  {s}: {c}/{total} ({100*c/total:.1f}%)")
print(f"\n  Full mission success (arrived): {episode_results.get('arrived', 0)}/{total}")
print(f"{'='*60}")

env.close()
simulation_app.close()
