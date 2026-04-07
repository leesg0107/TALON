"""Test: Can the drone physically grasp and lift a box from the ground?"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 100
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 60.0  # Long episode for full sequence
cfg.lock_gripper = False

# Box is now dynamic on pedestal (configured in env_cfg)
# No overrides needed — env_cfg has dynamic box on pedestal at z=0.54

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/best_models/best_fc34_loiter_tilt.pt")
agent.set_running_mode("eval")

num_envs = env.num_envs

# State machine per env
state = ["approach"] * num_envs  # approach → dock → grip → lift
align_count = torch.zeros(num_envs, dtype=torch.long, device=device)
grip_count = torch.zeros(num_envs, dtype=torch.long, device=device)
lift_start_z = torch.zeros(num_envs, device=device)

# Episode result tracking
from collections import Counter
episode_results = Counter()
max_lift_cm = []  # track max lift height per lift attempt

max_steps = 9000  # 60s at 150Hz
print(f"\n=== Grasp & Lift Test: {num_envs} envs ===\n")

obs, _ = env_wrapped.reset()

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    # Compute overlap
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
            # Gripper open, let policy approach
            action[i, 7] = 1.0
            if ov > 0.50:
                align_count[i] += 1
            # No reset: cumulative, same as training
            if align_count[i] > 300:  # cumulative ~2s
                state[i] = "dock"
                grip_count[i] = 0
                print(f"  env{i} step {step}: DOCKED (overlap={ov*100:.0f}%) → closing gripper")

        elif state[i] == "dock":
            # Close gripper firmly
            action[i, 7] = -1.0
            grip_count[i] += 1
            if grip_count[i] > 150:  # 1s to close fully
                state[i] = "lift"
                lift_start_z[i] = box_z
                print(f"  env{i} step {step}: GRIPPED → attempting lift (box_z={box_z:.3f})")

        elif state[i] == "lift":
            # Keep gripper closed, let policy fly (it will try to hover/ascend)
            action[i, 7] = -1.0
            # Override: add upward thrust bias to encourage lifting
            action[i, 2] = min(action[i, 2].item() + 0.3, 1.0)

            box_lifted = box_z - lift_start_z[i].item()
            if step % 150 == 0:
                print(f"  env{i} step {step}: LIFTING box_z={box_z:.3f} lift={box_lifted*100:.1f}cm drone_z={drone_z:.3f}")

            if box_lifted > 0.05:  # 5cm lift = SUCCESS
                state[i] = "success"
                print(f"  env{i} step {step}: *** LIFT SUCCESS *** box lifted {box_lifted*100:.1f}cm!")

        elif state[i] == "success":
            action[i, 7] = -1.0  # keep gripping

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    # Status every 20s
    if step % 3000 == 0:
        status = ""
        for i in range(num_envs):
            box_z = env.object_pos[i, 2].item()
            status += f"  env{i}:{state[i]}(bz={box_z:.3f})"
        print(f"  [{step/150:.0f}s]{status}")

    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        for i in reset_list:
            # Track max lift for lift/success states
            if state[i] in ("lift", "success"):
                lifted = env.object_pos[i, 2].item() - lift_start_z[i].item()
                max_lift_cm.append(lifted * 100)
            # Record best state reached this episode
            episode_results[state[i]] += 1
            state[i] = "approach"
            align_count[i] = 0
            grip_count[i] = 0
        obs, _ = env_wrapped.reset()

# Add remaining (not yet reset) envs
for i in range(num_envs):
    episode_results[state[i]] += 1

total_episodes = sum(episode_results.values())
print(f"\n=== RESULTS ===")
print(f"  Total episodes: {total_episodes}")
for s in ["success", "lift", "grip", "dock", "approach"]:
    c = episode_results.get(s, 0)
    if c > 0 or s in ["success", "approach"]:
        print(f"  {s}: {c}/{total_episodes} ({100*c/total_episodes:.1f}%)")
print(f"  Lift successes: {episode_results.get('success', 0)}/{total_episodes}")
if max_lift_cm:
    import numpy as np
    print(f"  Lift heights: mean={np.mean(max_lift_cm):.1f}cm  max={np.max(max_lift_cm):.1f}cm  min={np.min(max_lift_cm):.1f}cm")

env.close()
simulation_app.close()
