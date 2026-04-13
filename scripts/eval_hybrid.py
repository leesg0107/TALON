"""Hybrid (analytical + RL residual) evaluation — failure diagnosis."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 100
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 12.0
cfg.lock_gripper = True
cfg.residual_scale = 0.1  # RL residual: ±0.8 m/s² correction on top of analytical
# Dynamic box
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/stage3_residual/best_agent.pt")
agent.set_running_mode("eval")

num_envs = env.num_envs
episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
from envs.drone_env import quat_to_rot_matrix

max_eval_steps = 9000  # ~5 episodes per env at 12s

# ========================================================================
# Per-env tracking (reset on episode end)
# ========================================================================
env_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_min_xy = torch.full((num_envs,), float('inf'), device=device)
ep_min_xy_step = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_reached_5cm = torch.zeros(num_envs, dtype=torch.bool, device=device)
ep_reached_10cm = torch.zeros(num_envs, dtype=torch.bool, device=device)
ep_first_overlap_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
ep_dock_count = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_max_overlap = torch.zeros(num_envs, device=device)
ep_peak_dock_count = torch.zeros(num_envs, dtype=torch.long, device=device)

# First overlap snapshot state
ep_fo_xy_err = torch.zeros(num_envs, device=device)
ep_fo_vz = torch.zeros(num_envs, device=device)
ep_fo_tilt_deg = torch.zeros(num_envs, device=device)
ep_fo_z_offset = torch.zeros(num_envs, device=device)

# Box displacement tracking
ep_box_init_pos = torch.zeros(num_envs, 3, device=device)
ep_box_init_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)

# Initial XY error (for spawn distance analysis)
ep_init_xy_err = torch.zeros(num_envs, device=device)

# Oscillation
prev_xy_err = torch.full((num_envs,), float('inf'), device=device)
prev_xy_decreasing = torch.zeros(num_envs, dtype=torch.bool, device=device)
ep_oscillation = torch.zeros(num_envs, dtype=torch.long, device=device)

# Time series
SAMPLE_INTERVAL = 10
xy_err_time, z_offset_time, vz_time, tilt_time, overlap_time = [], [], [], [], []

# ========================================================================
# Episode-level results
# ========================================================================
results = {
    "min_xy_err": [], "min_xy_step": [],
    "reached_5cm": 0, "reached_10cm": 0,
    "first_overlap_step": [], "dock_success": 0, "total_episodes": 0,
    "xy_oscillation_count": [],
    "end_xy_err": [], "end_z_offset": [], "end_tilt_deg": [], "end_vz": [],
    "crashed": 0,
    # NEW: failure diagnosis
    "failure_category": [],     # "dock" / "overlap_lost" / "no_descent" / "far"
    "max_overlap": [],          # per-episode max overlap ratio
    "peak_dock_count": [],      # max consecutive overlap>0.5 steps
    "box_displacement": [],     # box XY displacement from initial pos
    "init_xy_err": [],          # initial XY error (spawn distance)
    # First overlap snapshot
    "fo_xy_err": [], "fo_vz": [], "fo_tilt_deg": [], "fo_z_offset": [],
}

print(f"\n=== Analytical Base Controller Eval (Failure Diagnosis) ===")
print(f"  Envs: {num_envs}, Steps: {max_eval_steps}, Episode: {episode_steps} steps")
print(f"  Box: dynamic\n")

obs, _ = env_wrapped.reset()

for step in range(max_eval_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_eval_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    env_step_count += 1

    # --- Compute metrics ---
    pos_w = env.robot.data.root_pos_w
    vel_w = env.robot.data.root_lin_vel_w
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)

    gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=device)
    gripper_pos_w = pos_w + torch.bmm(
        R, gripper_offset_b.expand(num_envs, 3).unsqueeze(-1)
    ).squeeze(-1)

    obj_pos = env.object_pos
    xy_err = torch.norm(gripper_pos_w[:, :2] - obj_pos[:, :2], dim=-1)
    z_offset = gripper_pos_w[:, 2] - (obj_pos[:, 2] + 0.04)
    vz = vel_w[:, 2]
    tilt_rad = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
    tilt_deg = tilt_rad * (180.0 / 3.14159)

    # Overlap
    box_offset_w = obj_pos - gripper_pos_w
    box_local = torch.bmm(R.transpose(1, 2), box_offset_w.unsqueeze(-1)).squeeze(-1)
    box_half = 0.04
    gripper_half_x, gripper_half_y = 0.05, 0.062
    overlap_x = (torch.min(torch.full_like(box_local[:, 0], gripper_half_x), box_local[:, 0] + box_half)
               - torch.max(torch.full_like(box_local[:, 0], -gripper_half_x), box_local[:, 0] - box_half)).clamp(min=0)
    overlap_y = (torch.min(torch.full_like(box_local[:, 1], gripper_half_y), box_local[:, 1] + box_half)
               - torch.max(torch.full_like(box_local[:, 1], -gripper_half_y), box_local[:, 1] - box_half)).clamp(min=0)
    z_in = ((box_local[:, 2] > -0.12) & (box_local[:, 2] < 0.02)).float()
    overlap_ratio = (overlap_x * overlap_y / (2 * box_half) ** 2) * z_in

    # --- Record initial state (step 1 of each episode) ---
    first_step_mask = (env_step_count == 1)
    if first_step_mask.any():
        ep_init_xy_err[first_step_mask] = xy_err[first_step_mask]
        ep_box_init_pos[first_step_mask] = obj_pos[first_step_mask]
        ep_box_init_recorded[first_step_mask] = True

    # --- Per-step tracking ---
    is_new_min = xy_err < ep_min_xy
    ep_min_xy = torch.where(is_new_min, xy_err, ep_min_xy)
    ep_min_xy_step = torch.where(is_new_min, env_step_count, ep_min_xy_step)
    ep_reached_5cm |= (xy_err < 0.05)
    ep_reached_10cm |= (xy_err < 0.10)

    # Max overlap
    ep_max_overlap = torch.max(ep_max_overlap, overlap_ratio)

    # Dock count (overlap > 0.5)
    ep_dock_count += (overlap_ratio > 0.5).long()
    ep_peak_dock_count = torch.max(ep_peak_dock_count, ep_dock_count)

    # First overlap snapshot
    ov_mask = (overlap_ratio > 0) & (ep_first_overlap_step < 0)
    if ov_mask.any():
        ep_first_overlap_step = torch.where(ov_mask, env_step_count, ep_first_overlap_step)
        ep_fo_xy_err[ov_mask] = xy_err[ov_mask]
        ep_fo_vz[ov_mask] = vz[ov_mask]
        ep_fo_tilt_deg[ov_mask] = tilt_deg[ov_mask]
        ep_fo_z_offset[ov_mask] = z_offset[ov_mask]

    # Oscillation
    xy_decreasing = xy_err < prev_xy_err
    sign_change = (xy_decreasing != prev_xy_decreasing) & (env_step_count > 2)
    ep_oscillation += sign_change.long()
    prev_xy_decreasing = xy_decreasing
    prev_xy_err = xy_err.clone()

    # Time series
    if step % SAMPLE_INTERVAL == 0:
        xy_err_time.append((step, xy_err.mean().item()))
        z_offset_time.append((step, z_offset.mean().item()))
        vz_time.append((step, vz.mean().item()))
        tilt_time.append((step, tilt_deg.mean().item()))
        overlap_time.append((step, overlap_ratio.mean().item()))

    # --- Episode end ---
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        for i in done_mask.nonzero(as_tuple=False).view(-1).tolist():
            results["total_episodes"] += 1
            results["min_xy_err"].append(ep_min_xy[i].item())
            results["min_xy_step"].append(int(ep_min_xy_step[i].item()))
            results["reached_5cm"] += int(ep_reached_5cm[i].item())
            results["reached_10cm"] += int(ep_reached_10cm[i].item())
            results["xy_oscillation_count"].append(int(ep_oscillation[i].item()))
            results["init_xy_err"].append(ep_init_xy_err[i].item())

            # End state
            results["end_xy_err"].append(xy_err[i].item())
            results["end_z_offset"].append(z_offset[i].item())
            results["end_tilt_deg"].append(tilt_deg[i].item())
            results["end_vz"].append(vz[i].item())

            if term_flat[i] and pos_w[i, 2].item() < 0.15:
                results["crashed"] += 1

            # Max overlap & peak dock count
            results["max_overlap"].append(ep_max_overlap[i].item())
            results["peak_dock_count"].append(int(ep_peak_dock_count[i].item()))

            # Box displacement
            if ep_box_init_recorded[i]:
                box_disp = torch.norm(obj_pos[i, :2] - ep_box_init_pos[i, :2]).item()
                results["box_displacement"].append(box_disp)
            else:
                results["box_displacement"].append(0.0)

            # First overlap snapshot
            if ep_first_overlap_step[i].item() >= 0:
                results["first_overlap_step"].append(int(ep_first_overlap_step[i].item()))
                results["fo_xy_err"].append(ep_fo_xy_err[i].item())
                results["fo_vz"].append(ep_fo_vz[i].item())
                results["fo_tilt_deg"].append(ep_fo_tilt_deg[i].item())
                results["fo_z_offset"].append(ep_fo_z_offset[i].item())

            # Dock success
            docked = ep_dock_count[i].item() >= 150
            if docked:
                results["dock_success"] += 1

            # --- Failure category ---
            had_overlap = ep_first_overlap_step[i].item() >= 0
            if docked:
                cat = "dock"
            elif had_overlap:
                cat = "overlap_lost"
            elif ep_reached_10cm[i]:
                cat = "no_descent"
            else:
                cat = "far"
            results["failure_category"].append(cat)

            # Reset per-env state
            env_step_count[i] = 0
            ep_min_xy[i] = float('inf')
            ep_min_xy_step[i] = 0
            ep_reached_5cm[i] = False
            ep_reached_10cm[i] = False
            ep_first_overlap_step[i] = -1
            ep_dock_count[i] = 0
            ep_max_overlap[i] = 0.0
            ep_peak_dock_count[i] = 0
            ep_fo_xy_err[i] = 0.0
            ep_fo_vz[i] = 0.0
            ep_fo_tilt_deg[i] = 0.0
            ep_fo_z_offset[i] = 0.0
            ep_box_init_recorded[i] = False
            ep_init_xy_err[i] = 0.0
            ep_oscillation[i] = 0
            prev_xy_err[i] = float('inf')
            prev_xy_decreasing[i] = False

        obs, _ = env_wrapped.reset()

    if step % 1000 == 0:
        print(f"  step {step:>5}  xy_err={xy_err.mean():.3f}  z_off={z_offset.mean():.3f}  "
              f"vz={vz.mean():.3f}  tilt={tilt_deg.mean():.1f}  ov={overlap_ratio.mean():.3f}")

# ========================================================================
# Results
# ========================================================================
import statistics as _stats

def _s(vals, fmt=".3f"):
    if not vals: return "(n=0)"
    m = sum(vals)/len(vals)
    md = _stats.median(vals)
    return f"mean={m:{fmt}} med={md:{fmt}} min={min(vals):{fmt}} max={max(vals):{fmt}} n={len(vals)}"

n_ep = max(results["total_episodes"], 1)
cats = results["failure_category"]
n_dock = cats.count("dock")
n_overlap_lost = cats.count("overlap_lost")
n_no_descent = cats.count("no_descent")
n_far = cats.count("far")

print(f"\n{'='*70}")
print(f"  ANALYTICAL BASE CONTROLLER — FAILURE DIAGNOSIS")
print(f"{'='*70}")
print(f"  Total episodes: {n_ep}")
print(f"  Crash: {results['crashed']}/{n_ep} ({100*results['crashed']/n_ep:.0f}%)")

print(f"\n  ==================== FAILURE BREAKDOWN ====================")
print(f"  [DOCK]         {n_dock:>4}/{n_ep} ({100*n_dock/n_ep:5.1f}%)  ← success")
print(f"  [OVERLAP_LOST] {n_overlap_lost:>4}/{n_ep} ({100*n_overlap_lost/n_ep:5.1f}%)  ← got overlap, lost dock")
print(f"  [NO_DESCENT]   {n_no_descent:>4}/{n_ep} ({100*n_no_descent/n_ep:5.1f}%)  ← reached <10cm, no overlap")
print(f"  [FAR]          {n_far:>4}/{n_ep} ({100*n_far/n_ep:5.1f}%)  ← never reached <10cm")

# --- Per-category analysis ---
def _cat_vals(key, cat):
    return [results[key][i] for i, c in enumerate(cats) if c == cat]

print(f"\n  ==================== PER-CATEGORY DETAIL ====================")

# FAR episodes
if n_far > 0:
    far_init = _cat_vals("init_xy_err", "far")
    far_min_xy = _cat_vals("min_xy_err", "far")
    far_osc = _cat_vals("xy_oscillation_count", "far")
    print(f"\n  --- FAR (never <10cm): {n_far} episodes ---")
    print(f"  Init XY err:     {_s(far_init)}")
    print(f"  Min XY err:      {_s(far_min_xy)}")
    print(f"  XY oscillation:  {_s(far_osc, '.0f')}")
    far_high_osc = sum(1 for o in far_osc if o > 20)
    print(f"  High osc (>20):  {far_high_osc}/{n_far} ({100*far_high_osc/n_far:.0f}%)")

# NO_DESCENT episodes
if n_no_descent > 0:
    nd_init = _cat_vals("init_xy_err", "no_descent")
    nd_min_xy = _cat_vals("min_xy_err", "no_descent")
    nd_end_z = _cat_vals("end_z_offset", "no_descent")
    nd_max_ov = _cat_vals("max_overlap", "no_descent")
    print(f"\n  --- NO_DESCENT (reached <10cm, no overlap): {n_no_descent} episodes ---")
    print(f"  Init XY err:     {_s(nd_init)}")
    print(f"  Min XY err:      {_s(nd_min_xy)}")
    print(f"  End z_offset:    {_s(nd_end_z)}")
    print(f"  Max overlap:     {_s(nd_max_ov)}")

# OVERLAP_LOST episodes
if n_overlap_lost > 0:
    ol_max_ov = _cat_vals("max_overlap", "overlap_lost")
    ol_peak_dc = _cat_vals("peak_dock_count", "overlap_lost")
    ol_box_disp = _cat_vals("box_displacement", "overlap_lost")
    ol_end_xy = _cat_vals("end_xy_err", "overlap_lost")
    ol_end_z = _cat_vals("end_z_offset", "overlap_lost")
    ol_end_tilt = _cat_vals("end_tilt_deg", "overlap_lost")
    # First overlap snapshot (only for episodes that HAD overlap)
    ol_indices = [i for i, c in enumerate(cats) if c == "overlap_lost"]
    # fo_ lists have different indexing (only episodes with overlap)
    # Re-collect from fo results by matching
    ol_fo_vz = []
    ol_fo_tilt = []
    ol_fo_xy = []
    fo_idx = 0
    for i, c in enumerate(cats):
        if results["max_overlap"][i] > 0 or c in ("dock", "overlap_lost"):
            if c == "overlap_lost" and fo_idx < len(results["fo_vz"]):
                ol_fo_vz.append(results["fo_vz"][fo_idx])
                ol_fo_tilt.append(results["fo_tilt_deg"][fo_idx])
                ol_fo_xy.append(results["fo_xy_err"][fo_idx])
            if i < len(cats) and (c == "dock" or c == "overlap_lost"):
                fo_idx += 1

    print(f"\n  --- OVERLAP_LOST (got overlap, failed dock): {n_overlap_lost} episodes ---")
    print(f"  Max overlap:       {_s(ol_max_ov)}")
    print(f"  Peak dock count:   {_s(ol_peak_dc, '.0f')}  (need 150 for success)")
    near_miss = sum(1 for d in ol_peak_dc if d >= 100)
    some_dock = sum(1 for d in ol_peak_dc if 10 <= d < 100)
    brief_touch = sum(1 for d in ol_peak_dc if d < 10)
    print(f"    Near miss (≥100): {near_miss}  Some dock (10-99): {some_dock}  Brief touch (<10): {brief_touch}")
    print(f"  Box displacement:  {_s(ol_box_disp)}")
    box_moved = sum(1 for d in ol_box_disp if d > 0.02)
    print(f"    Box moved >2cm:  {box_moved}/{n_overlap_lost} ({100*box_moved/n_overlap_lost:.0f}%)")
    print(f"  End XY err:        {_s(ol_end_xy)}")
    print(f"  End z_offset:      {_s(ol_end_z)}")
    print(f"  End tilt:          {_s(ol_end_tilt, '.1f')}")

# DOCK episodes
if n_dock > 0:
    dk_box_disp = _cat_vals("box_displacement", "dock")
    dk_init = _cat_vals("init_xy_err", "dock")
    print(f"\n  --- DOCK (success): {n_dock} episodes ---")
    print(f"  Init XY err:       {_s(dk_init)}")
    print(f"  Box displacement:  {_s(dk_box_disp)}")

print(f"\n  ==================== SPAWN DISTANCE ANALYSIS ====================")
print(f"  All episodes init XY: {_s(results['init_xy_err'])}")
# Dock rate by spawn distance bucket
buckets = [(0, 0.5, "0-50cm"), (0.5, 1.0, "50cm-1m"), (1.0, 2.0, "1-2m"), (2.0, 5.0, "2-5m"), (5.0, 100, ">5m")]
print(f"  {'Spawn range':>12}  {'Episodes':>8}  {'Dock%':>6}  {'Far%':>6}")
for lo, hi, label in buckets:
    idx = [i for i, d in enumerate(results["init_xy_err"]) if lo <= d < hi]
    if not idx:
        continue
    n = len(idx)
    d = sum(1 for i in idx if cats[i] == "dock")
    f = sum(1 for i in idx if cats[i] == "far")
    print(f"  {label:>12}  {n:>8}  {100*d/n:>5.1f}%  {100*f/n:>5.1f}%")

print(f"\n  ==================== BOX DISPLACEMENT ====================")
print(f"  All episodes: {_s(results['box_displacement'])}")
disp_buckets = [(0, 0.01, "<1cm"), (0.01, 0.03, "1-3cm"), (0.03, 0.10, "3-10cm"), (0.10, 100, ">10cm")]
for lo, hi, label in disp_buckets:
    n = sum(1 for d in results["box_displacement"] if lo <= d < hi)
    print(f"  {label:>8}: {n}/{n_ep} ({100*n/n_ep:.0f}%)")

print(f"\n  ==================== FIRST OVERLAP SNAPSHOT ====================")
if results["fo_vz"]:
    print(f"  XY err at first overlap: {_s(results['fo_xy_err'])}")
    print(f"  v_z at first overlap:    {_s(results['fo_vz'])}")
    print(f"  Tilt at first overlap:   {_s(results['fo_tilt_deg'], '.1f')}")
    print(f"  z_offset at 1st overlap: {_s(results['fo_z_offset'])}")

print(f"\n  ==================== TIMELINE ====================")
print(f"  {'step':>6}  {'xy_err':>8}  {'z_offset':>9}  {'v_z':>7}  {'tilt':>6}  {'overlap':>8}")
for i in range(0, len(xy_err_time), max(1, len(xy_err_time)//15)):
    s, xy = xy_err_time[i]
    _, zo = z_offset_time[i]
    _, vz_v = vz_time[i]
    _, ti = tilt_time[i]
    _, ov = overlap_time[i]
    print(f"  {s:>6}  {xy:>8.3f}  {zo:>9.3f}  {vz_v:>+7.3f}  {ti:>6.1f}  {ov:>8.4f}")

print(f"{'='*70}")

env.close()
simulation_app.close()
