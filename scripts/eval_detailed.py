"""Comprehensive trajectory-level failure analysis."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import numpy as np
from collections import defaultdict
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 100
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 40.0
cfg.lock_gripper = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/best_models/best_col_strong.pt")
agent.set_running_mode("eval")

num_envs = env.num_envs

# Per-env tracking
align_count = torch.zeros(num_envs, dtype=torch.long, device=device)
gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)
docked = torch.zeros(num_envs, dtype=torch.bool, device=device)
env_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

# Per-env trajectory recording
max_overlap = torch.zeros(num_envs, device=device)
min_xy_err = torch.full((num_envs,), 10.0, device=device)
max_speed = torch.zeros(num_envs, device=device)
time_near = torch.zeros(num_envs, dtype=torch.long, device=device)
time_above = torch.zeros(num_envs, dtype=torch.long, device=device)
time_overlap_50 = torch.zeros(num_envs, dtype=torch.long, device=device)
collision_count = torch.zeros(num_envs, dtype=torch.long, device=device)
# Track direction: did drone overshoot past box?
overshot_xy = torch.zeros(num_envs, dtype=torch.bool, device=device)
# Track if drone ever reached xy<0.15m
reached_close = torch.zeros(num_envs, dtype=torch.bool, device=device)
# Reward accumulation per env
sum_r_approach = torch.zeros(num_envs, device=device)
sum_r_contain = torch.zeros(num_envs, device=device)
sum_r_fine = torch.zeros(num_envs, device=device)
sum_r_column = torch.zeros(num_envs, device=device)
sum_column_score = torch.zeros(num_envs, device=device)
max_column_score = torch.zeros(num_envs, device=device)
time_in_column = torch.zeros(num_envs, dtype=torch.long, device=device)
# Track initial approach direction
initial_xy_err = torch.zeros(num_envs, device=device)
# Speed at closest approach
speed_at_min_xy = torch.zeros(num_envs, device=device)
# Previous xy for overshoot detection
prev_xy_err = torch.full((num_envs,), 10.0, device=device)
approaching = torch.ones(num_envs, dtype=torch.bool, device=device)  # True until first close approach

# Episode records
all_episodes = []

max_eval_steps = 12000
print(f"\n=== Detailed Trajectory Analysis: {num_envs} envs x {max_eval_steps} steps ===\n")

obs, _ = env_wrapped.reset()

# Record initial xy_err
pos_w = env.robot.data.root_pos_w
gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=device)
R = quat_to_rot_matrix(env.robot.data.root_quat_w)
gripper_pos_w = pos_w + torch.bmm(R, gripper_offset_b.expand(num_envs, -1).unsqueeze(-1)).squeeze(-1)
initial_xy_err[:] = torch.norm(gripper_pos_w[:, :2] - env.object_pos[:, :2], dim=-1)

for step in range(max_eval_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_eval_steps)[0]

    # Compute metrics
    pos_w = env.robot.data.root_pos_w
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
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

    xy_err = torch.norm(gripper_pos_w[:, :2] - env.object_pos[:, :2], dim=-1)
    z_err = torch.abs(gripper_pos_w[:, 2] - env.object_pos[:, 2])
    pos_err = torch.norm(gripper_pos_w - env.object_pos, dim=-1)

    # Column containment
    column_half_x, column_half_y = 0.05, 0.104
    x_contain = (1.0 - torch.abs(box_local[:, 0]) / column_half_x).clamp(0.0, 1.0)
    y_contain = (1.0 - torch.abs(box_local[:, 1]) / column_half_y).clamp(0.0, 1.0)
    z_below = (box_local[:, 2] < 0.02).float()
    column_score = x_contain * y_contain * z_below
    vel_w = env.robot.data.root_lin_vel_w
    speed = torch.norm(vel_w, dim=-1)
    gripper_above = gripper_pos_w[:, 2] > env.object_pos[:, 2]
    drone_z = pos_w[:, 2]

    # Compute rewards (mirror training reward computation)
    approach_gate = ((0.25 - xy_err) / 0.20).clamp(0.0, 1.0)
    r_approach_val = 4.0 * torch.exp(-1.2 * pos_err.clamp(min=0.3))
    r_contain_val = 15.0 * overlap_ratio + 10.0 * (overlap_ratio > 0.90).float()
    r_fine_val = 5.0 * torch.exp(-10.0 * xy_err) * approach_gate
    r_column_val = (1.5 * x_contain + 1.5 * y_contain + 2.0 * z_below) * approach_gate

    # Accumulate per-env rewards
    sum_r_approach += r_approach_val
    sum_r_contain += r_contain_val
    sum_r_fine += r_fine_val
    sum_r_column += r_column_val
    sum_column_score += column_score
    max_column_score = torch.max(max_column_score, column_score)
    time_in_column += (column_score > 0.3).long()

    # Update per-env stats
    max_overlap = torch.max(max_overlap, overlap_ratio)
    # Track speed at min xy
    closer = xy_err < min_xy_err
    speed_at_min_xy = torch.where(closer, speed, speed_at_min_xy)
    min_xy_err = torch.min(min_xy_err, xy_err)
    max_speed = torch.max(max_speed, speed)
    time_near += (xy_err < 0.5).long()
    time_above += gripper_above.long()
    time_overlap_50 += (overlap_ratio > 0.5).long()
    reached_close |= (xy_err < 0.15)

    # Overshoot detection: xy was decreasing, now increasing past box
    getting_farther = (xy_err > prev_xy_err + 0.02) & (prev_xy_err < 0.3)
    overshot_xy |= getting_farther & approaching
    approaching &= ~(xy_err < 0.15)  # stop tracking after first close approach
    prev_xy_err = xy_err.clone()

    # Collision detection: sudden speed increase near box
    near_box = xy_err < 0.3
    fast_change = speed > 1.0
    collision_count += (near_box & fast_change).long()

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

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    env_step_count += 1

    # Episode reset
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        for i in reset_list:
            is_docked = docked[i].item()
            d_z = drone_z[i].item()
            b_z = env.object_pos[i, 2].item()
            xy = xy_err[i].item()
            is_term = term_flat[i].item()

            if is_term and d_z < 0.15:
                fail_type = "crash"
            elif not is_docked and xy > 0.5:
                fail_type = "far"
            elif not is_docked and xy <= 0.5 and d_z < b_z:
                fail_type = "below"
            elif not is_docked:
                fail_type = "near_timeout"
            else:
                fail_type = "success"

            steps = max(env_step_count[i].item(), 1)
            ep = {
                "result": fail_type,
                "max_overlap": max_overlap[i].item(),
                "min_xy_err": min_xy_err[i].item(),
                "final_xy_err": xy,
                "initial_xy_err": initial_xy_err[i].item(),
                "speed_at_min_xy": speed_at_min_xy[i].item(),
                "max_speed": max_speed[i].item(),
                "time_near_pct": time_near[i].item() / steps * 100,
                "time_above_pct": time_above[i].item() / steps * 100,
                "time_overlap50_pct": time_overlap_50[i].item() / steps * 100,
                "reached_close": reached_close[i].item(),
                "overshot": overshot_xy[i].item(),
                "collisions": collision_count[i].item(),
                "episode_steps": steps,
                "final_drone_z": d_z,
                "avg_r_approach": sum_r_approach[i].item() / steps,
                "avg_r_contain": sum_r_contain[i].item() / steps,
                "avg_r_fine": sum_r_fine[i].item() / steps,
                "avg_r_column": sum_r_column[i].item() / steps,
                "avg_column_score": sum_column_score[i].item() / steps,
                "max_column_score": max_column_score[i].item(),
                "time_in_column_pct": time_in_column[i].item() / steps * 100,
            }
            all_episodes.append(ep)

            # Reset
            align_count[i] = 0
            gripper_closed[i] = False
            docked[i] = False
            max_overlap[i] = 0
            min_xy_err[i] = 10.0
            max_speed[i] = 0
            speed_at_min_xy[i] = 0
            time_near[i] = 0
            time_above[i] = 0
            time_overlap_50[i] = 0
            collision_count[i] = 0
            overshot_xy[i] = False
            reached_close[i] = False
            approaching[i] = True
            prev_xy_err[i] = 10.0
            env_step_count[i] = 0
            sum_r_approach[i] = 0
            sum_r_contain[i] = 0
            sum_r_fine[i] = 0
            sum_r_column[i] = 0
            sum_column_score[i] = 0
            max_column_score[i] = 0
            time_in_column[i] = 0

        obs, _ = env_wrapped.reset()
        # Update initial_xy_err for reset envs
        pos_w = env.robot.data.root_pos_w
        R = quat_to_rot_matrix(env.robot.data.root_quat_w)
        gripper_pos_w = pos_w + torch.bmm(R, gripper_offset_b.expand(num_envs, -1).unsqueeze(-1)).squeeze(-1)
        for i in reset_list:
            initial_xy_err[i] = torch.norm(gripper_pos_w[i, :2] - env.object_pos[i, :2]).item()

    if step % 3000 == 0:
        n_done = len(all_episodes)
        n_succ = sum(1 for e in all_episodes if e["result"] == "success")
        print(f"  step {step}: {n_done} episodes, {n_succ} success")

# === ANALYSIS ===
print(f"\n{'='*70}")
print(f"  COMPREHENSIVE TRAJECTORY ANALYSIS")
print(f"{'='*70}")

total = len(all_episodes)
by_type = defaultdict(list)
for ep in all_episodes:
    by_type[ep["result"]].append(ep)

print(f"\n  Total episodes: {total}")
for t in ["success", "near_timeout", "far", "below", "crash"]:
    eps = by_type[t]
    n = len(eps)
    print(f"\n  {'='*60}")
    print(f"  {t.upper()}: {n}/{total} ({100*n/max(total,1):.1f}%)")
    if n == 0:
        continue

    max_ovs = [e["max_overlap"] for e in eps]
    min_xys = [e["min_xy_err"] for e in eps]
    final_xys = [e["final_xy_err"] for e in eps]
    speeds_at_min = [e["speed_at_min_xy"] for e in eps]
    max_speeds = [e["max_speed"] for e in eps]
    near_pcts = [e["time_near_pct"] for e in eps]
    above_pcts = [e["time_above_pct"] for e in eps]
    ov50_pcts = [e["time_overlap50_pct"] for e in eps]
    reached = sum(1 for e in eps if e["reached_close"])
    overshot = sum(1 for e in eps if e["overshot"])
    collisions = [e["collisions"] for e in eps]

    print(f"    Max overlap:     mean={np.mean(max_ovs)*100:.1f}%  median={np.median(max_ovs)*100:.1f}%")
    print(f"    Min XY err:      mean={np.mean(min_xys)*100:.1f}cm  median={np.median(min_xys)*100:.1f}cm")
    print(f"    Final XY err:    mean={np.mean(final_xys)*100:.1f}cm  median={np.median(final_xys)*100:.1f}cm")
    print(f"    Speed at min XY: mean={np.mean(speeds_at_min):.2f}m/s  median={np.median(speeds_at_min):.2f}m/s")
    print(f"    Max speed:       mean={np.mean(max_speeds):.2f}m/s  median={np.median(max_speeds):.2f}m/s")
    print(f"    Time near(<0.5m):  mean={np.mean(near_pcts):.1f}%")
    print(f"    Time above box:    mean={np.mean(above_pcts):.1f}%")
    print(f"    Time overlap>50%:  mean={np.mean(ov50_pcts):.1f}%")
    print(f"    Reached xy<0.15m:  {reached}/{n} ({100*reached/n:.0f}%)")
    print(f"    Overshot past box: {overshot}/{n} ({100*overshot/n:.0f}%)")
    print(f"    Collisions (fast near box): mean={np.mean(collisions):.1f}")

    # Spawn condition for all types
    init_xys = [e["initial_xy_err"] for e in eps]
    print(f"    Initial XY err:  mean={np.mean(init_xys)*100:.1f}cm  median={np.median(init_xys)*100:.1f}cm  min={np.min(init_xys)*100:.1f}cm  max={np.max(init_xys)*100:.1f}cm")

    if t == "far":
        went_wrong = sum(1 for e in eps if e["final_xy_err"] > e["initial_xy_err"])
        print(f"    Got farther than start: {went_wrong}/{n} ({100*went_wrong/n:.0f}%)")

    if t == "near_timeout":
        almost = sum(1 for o in max_ovs if o > 0.5)
        partial = sum(1 for o in max_ovs if 0.1 < o <= 0.5)
        never = sum(1 for o in max_ovs if o <= 0.1)
        print(f"    Sub-categories:")
        print(f"      Almost docked (ov>50%):  {almost} ({100*almost/n:.0f}%)")
        print(f"      Partial (10-50%):        {partial} ({100*partial/n:.0f}%)")
        print(f"      Never approached (<10%): {never} ({100*never/n:.0f}%)")

# === SUCCESS vs FAILURE comparison ===
if by_type["success"] and by_type["near_timeout"]:
    print(f"\n  {'='*60}")
    print(f"  SUCCESS vs NEAR_TIMEOUT COMPARISON")
    print(f"  {'='*60}")
    succ = by_type["success"]
    fail = by_type["near_timeout"]
    print(f"    {'Metric':<25} {'Success':>12} {'Near Timeout':>12}")
    print(f"    {'-'*49}")
    for metric, fmt in [
        ("initial_xy_err", ".4f"),
        ("speed_at_min_xy", ".2f"),
        ("max_speed", ".2f"),
        ("min_xy_err", ".4f"),
        ("max_overlap", ".2f"),
        ("avg_r_approach", ".2f"),
        ("avg_r_contain", ".2f"),
        ("avg_r_fine", ".2f"),
        ("avg_r_column", ".2f"),
        ("avg_column_score", ".3f"),
        ("max_column_score", ".3f"),
        ("time_in_column_pct", ".1f"),
        ("collisions", ".1f"),
    ]:
        s_val = np.mean([e[metric] for e in succ])
        f_val = np.mean([e[metric] for e in fail])
        print(f"    {metric:<25} {s_val:>12{fmt}} {f_val:>12{fmt}}")

print(f"\n{'='*70}")

env.close()
simulation_app.close()
