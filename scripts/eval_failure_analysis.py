"""Detailed failure analysis for dock evaluation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 100
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 40.0  # Full length for analysis
cfg.lock_gripper = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/best_models/best_fc10_saturate.pt")
agent.set_running_mode("eval")

num_envs = env.num_envs

# Per-env tracking
align_count = torch.zeros(num_envs, dtype=torch.long, device=device)
gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)
docked = torch.zeros(num_envs, dtype=torch.bool, device=device)
env_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

# Per-env episode stats (reset each episode)
max_overlap = torch.zeros(num_envs, device=device)
min_xy_err = torch.full((num_envs,), 10.0, device=device)
time_in_column = torch.zeros(num_envs, dtype=torch.long, device=device)  # steps where column_score > 0.3
time_near = torch.zeros(num_envs, dtype=torch.long, device=device)  # steps where xy < 0.5m
ever_above_box = torch.zeros(num_envs, dtype=torch.bool, device=device)  # ever had gripper above box

# Aggregate results
dock_success = 0
total_episodes = 0
near_timeout_details = []  # List of dicts for each near-timeout failure

max_eval_steps = 12000
print(f"\n=== Failure Analysis: {num_envs} envs x {max_eval_steps} steps ===\n")

obs, _ = env_wrapped.reset()

for step in range(max_eval_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_eval_steps)[0]

    # Compute gripper position and overlap
    pos_w = env.robot.data.root_pos_w
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=device)
    gripper_pos_w = pos_w + torch.bmm(R, gripper_offset_b.expand(num_envs, -1).unsqueeze(-1)).squeeze(-1)

    box_offset_w = env.object_pos - gripper_pos_w
    box_local = torch.bmm(R.transpose(1, 2), box_offset_w.unsqueeze(-1)).squeeze(-1)

    box_half = 0.04
    gripper_half_x, gripper_half_y = 0.05, 0.062
    column_half_x, column_half_y = 0.05, 0.104

    # Overlap
    overlap_x = (torch.min(torch.full_like(box_local[:, 0], gripper_half_x), box_local[:, 0] + box_half)
               - torch.max(torch.full_like(box_local[:, 0], -gripper_half_x), box_local[:, 0] - box_half)).clamp(min=0.0)
    overlap_y = (torch.min(torch.full_like(box_local[:, 1], gripper_half_y), box_local[:, 1] + box_half)
               - torch.max(torch.full_like(box_local[:, 1], -gripper_half_y), box_local[:, 1] - box_half)).clamp(min=0.0)
    box_area = (2 * box_half) ** 2
    overlap_xy = (overlap_x * overlap_y) / box_area
    z_in_range = ((box_local[:, 2] > -0.12) & (box_local[:, 2] < 0.02)).float()
    overlap_ratio = overlap_xy * z_in_range

    xy_err = torch.norm(gripper_pos_w[:, :2] - env.object_pos[:, :2], dim=-1)

    # Column score
    x_contain = (1.0 - torch.abs(box_local[:, 0]) / column_half_x).clamp(0.0, 1.0)
    y_contain = (1.0 - torch.abs(box_local[:, 1]) / column_half_y).clamp(0.0, 1.0)
    z_below = (box_local[:, 2] < 0.02).float()
    column_score = x_contain * y_contain * z_below

    # Update per-env stats
    max_overlap = torch.max(max_overlap, overlap_ratio)
    min_xy_err = torch.min(min_xy_err, xy_err)
    time_in_column += (column_score > 0.3).long()
    time_near += (xy_err < 0.5).long()
    gripper_above = (gripper_pos_w[:, 2] > env.object_pos[:, 2])
    ever_above_box |= gripper_above

    # Gripper control
    for i in range(num_envs):
        ov = overlap_ratio[i].item()
        if gripper_closed[i]:
            action[i, 7] = -1.0
        else:
            action[i, 7] = 1.0
            if ov > 0.85:
                align_count[i] += 1
            else:
                align_count[i] = 0
            if align_count[i] > 20:
                action[i, 7] = -1.0
                gripper_closed[i] = True
                if not docked[i]:
                    docked[i] = True
                    dock_success += 1

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    env_step_count += 1

    # Episode reset + detailed failure logging
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        total_episodes += len(reset_list)
        for i in reset_list:
            is_docked = docked[i].item()
            if not is_docked:
                drone_z = pos_w[i, 2].item()
                box_z = env.object_pos[i, 2].item()
                xy = xy_err[i].item()
                is_term = term_flat[i].item()

                # Classify
                if is_term and drone_z < 0.15:
                    fail_type = "crash"
                elif xy > 0.5:
                    fail_type = "far"
                elif xy <= 0.5 and drone_z < box_z:
                    fail_type = "below"
                else:
                    fail_type = "near_timeout"

                # Detailed info for near_timeout
                if fail_type == "near_timeout":
                    near_timeout_details.append({
                        "max_overlap": max_overlap[i].item(),
                        "min_xy_err": min_xy_err[i].item(),
                        "final_xy_err": xy,
                        "final_drone_z": drone_z,
                        "final_box_z": box_z,
                        "time_in_column_pct": time_in_column[i].item() / max(env_step_count[i].item(), 1) * 100,
                        "time_near_pct": time_near[i].item() / max(env_step_count[i].item(), 1) * 100,
                        "ever_above": ever_above_box[i].item(),
                        "episode_steps": env_step_count[i].item(),
                    })

            # Reset per-env stats
            align_count[i] = 0
            gripper_closed[i] = False
            docked[i] = False
            max_overlap[i] = 0
            min_xy_err[i] = 10.0
            time_in_column[i] = 0
            time_near[i] = 0
            ever_above_box[i] = False
            env_step_count[i] = 0
        obs, _ = env_wrapped.reset()

# === Results ===
print(f"\n{'='*70}")
print(f"  EVALUATION + FAILURE ANALYSIS")
print(f"{'='*70}")
print(f"  Total episodes:  {total_episodes}")
print(f"  Dock success:    {dock_success}/{total_episodes} ({100*dock_success/max(total_episodes,1):.1f}%)")
print(f"  Near timeout:    {len(near_timeout_details)} cases")

if near_timeout_details:
    max_ovs = [d["max_overlap"] for d in near_timeout_details]
    min_xys = [d["min_xy_err"] for d in near_timeout_details]
    final_xys = [d["final_xy_err"] for d in near_timeout_details]
    col_pcts = [d["time_in_column_pct"] for d in near_timeout_details]
    near_pcts = [d["time_near_pct"] for d in near_timeout_details]
    above_pct = sum(1 for d in near_timeout_details if d["ever_above"]) / len(near_timeout_details) * 100

    print(f"\n  --- Near Timeout Detail ({len(near_timeout_details)} cases) ---")
    print(f"  Max overlap ever reached:")
    print(f"    mean={np.mean(max_ovs)*100:.1f}%  median={np.median(max_ovs)*100:.1f}%  max={np.max(max_ovs)*100:.1f}%")
    print(f"    >80%: {sum(1 for o in max_ovs if o>0.8)}/{len(max_ovs)} ({100*sum(1 for o in max_ovs if o>0.8)/len(max_ovs):.0f}%)")
    print(f"    >50%: {sum(1 for o in max_ovs if o>0.5)}/{len(max_ovs)} ({100*sum(1 for o in max_ovs if o>0.5)/len(max_ovs):.0f}%)")
    print(f"    >20%: {sum(1 for o in max_ovs if o>0.2)}/{len(max_ovs)} ({100*sum(1 for o in max_ovs if o>0.2)/len(max_ovs):.0f}%)")
    print(f"    =0%:  {sum(1 for o in max_ovs if o<0.01)}/{len(max_ovs)}")

    print(f"  Min XY err ever reached:")
    print(f"    mean={np.mean(min_xys)*100:.1f}cm  median={np.median(min_xys)*100:.1f}cm  min={np.min(min_xys)*100:.1f}cm")

    print(f"  Final XY err (at timeout):")
    print(f"    mean={np.mean(final_xys)*100:.1f}cm  median={np.median(final_xys)*100:.1f}cm")

    print(f"  Time in column (column_score>0.3):")
    print(f"    mean={np.mean(col_pcts):.1f}%  median={np.median(col_pcts):.1f}%")

    print(f"  Time near box (xy<0.5m):")
    print(f"    mean={np.mean(near_pcts):.1f}%  median={np.median(near_pcts):.1f}%")

    print(f"  Ever had gripper above box: {above_pct:.0f}%")

    # Subcategorize near_timeout
    almost = sum(1 for o in max_ovs if o > 0.5)
    partial = sum(1 for o in max_ovs if 0.1 < o <= 0.5)
    never = sum(1 for o in max_ovs if o <= 0.1)
    print(f"\n  Near Timeout Sub-categories:")
    print(f"    Almost docked (max_ov>50%): {almost} ({100*almost/len(max_ovs):.0f}%) ← close, needs stability")
    print(f"    Partial approach (10-50%):  {partial} ({100*partial/len(max_ovs):.0f}%) ← approached but misaligned")
    print(f"    Never approached (<10%):    {never} ({100*never/len(max_ovs):.0f}%) ← stuck/wandering")

print(f"{'='*70}")

env.close()
simulation_app.close()
