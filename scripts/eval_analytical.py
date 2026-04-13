"""Analytical base controller evaluation — PD gain tuning."""
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
cfg.episode_length_s = 8.0
cfg.lock_gripper = True
# Dynamic box
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
# Agent needed for wrapper but output is ignored (analytical-only mode)
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/stage3_dynamic_v2/best_agent.pt")
agent.set_running_mode("eval")

num_envs = env.num_envs
episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
from envs.drone_env import quat_to_rot_matrix

# --- Per-step accumulators ---
max_eval_steps = 6000  # ~5 episodes per env
num_episodes = [0]

# Per-env tracking
env_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_min_xy = torch.full((num_envs,), float('inf'), device=device)
ep_min_xy_step = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_reached_5cm = torch.zeros(num_envs, dtype=torch.bool, device=device)
ep_reached_10cm = torch.zeros(num_envs, dtype=torch.bool, device=device)
ep_first_overlap_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
ep_dock_count = torch.zeros(num_envs, dtype=torch.long, device=device)  # overlap>0.5 cumulative

# Time series for XY convergence analysis (sample every 10 steps)
SAMPLE_INTERVAL = 10
xy_err_time = []  # list of (step, mean_xy_err)
z_offset_time = []
vz_time = []
tilt_time = []
overlap_time = []

# Episode-level results
results = {
    "min_xy_err": [],
    "min_xy_step": [],
    "reached_5cm": 0,
    "reached_10cm": 0,
    "first_overlap_step": [],
    "dock_success": 0,
    "total_episodes": 0,
    # At first overlap
    "fo_xy_err": [],
    "fo_vz": [],
    "fo_tilt_deg": [],
    "fo_z_offset": [],
    # XY oscillation: count of sign changes in xy_err derivative
    "xy_oscillation_count": [],
    # Terminal state
    "end_xy_err": [],
    "end_z_offset": [],
    "end_tilt_deg": [],
    "end_vz": [],
    "crashed": 0,
}

# For oscillation detection
prev_xy_err = torch.full((num_envs,), float('inf'), device=device)
prev_xy_decreasing = torch.zeros(num_envs, dtype=torch.bool, device=device)
ep_oscillation = torch.zeros(num_envs, dtype=torch.long, device=device)

print(f"\n=== Analytical Base Controller Eval ===")
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

    # Overlap (same as training)
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

    # --- Per-step tracking ---
    is_new_min = xy_err < ep_min_xy
    ep_min_xy = torch.where(is_new_min, xy_err, ep_min_xy)
    ep_min_xy_step = torch.where(is_new_min, env_step_count, ep_min_xy_step)
    ep_reached_5cm |= (xy_err < 0.05)
    ep_reached_10cm |= (xy_err < 0.10)

    # First overlap
    ov_mask = (overlap_ratio > 0) & (ep_first_overlap_step < 0)
    ep_first_overlap_step = torch.where(ov_mask, env_step_count, ep_first_overlap_step)

    # Dock count (overlap > 0.5)
    ep_dock_count += (overlap_ratio > 0.5).long()

    # XY oscillation detection (sign change in Δxy)
    xy_decreasing = xy_err < prev_xy_err
    sign_change = (xy_decreasing != prev_xy_decreasing) & (env_step_count > 2)
    ep_oscillation += sign_change.long()
    prev_xy_decreasing = xy_decreasing
    prev_xy_err = xy_err.clone()

    # Time series sampling
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
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        for i in reset_list:
            results["total_episodes"] += 1
            results["min_xy_err"].append(ep_min_xy[i].item())
            results["min_xy_step"].append(int(ep_min_xy_step[i].item()))
            results["reached_5cm"] += int(ep_reached_5cm[i].item())
            results["reached_10cm"] += int(ep_reached_10cm[i].item())
            results["xy_oscillation_count"].append(int(ep_oscillation[i].item()))

            # End state
            results["end_xy_err"].append(xy_err[i].item())
            results["end_z_offset"].append(z_offset[i].item())
            results["end_tilt_deg"].append(tilt_deg[i].item())
            results["end_vz"].append(vz[i].item())

            if term_flat[i] and pos_w[i, 2].item() < 0.15:
                results["crashed"] += 1

            # First overlap snapshot
            if ep_first_overlap_step[i].item() >= 0:
                results["first_overlap_step"].append(int(ep_first_overlap_step[i].item()))
                # Note: we can't retroactively get the state at first overlap
                # but we record the step for timing analysis

            # Dock success (cumulative overlap>0.5 for 150+ steps)
            if ep_dock_count[i].item() >= 150:
                results["dock_success"] += 1

            # Reset per-env state
            env_step_count[i] = 0
            ep_min_xy[i] = float('inf')
            ep_min_xy_step[i] = 0
            ep_reached_5cm[i] = False
            ep_reached_10cm[i] = False
            ep_first_overlap_step[i] = -1
            ep_dock_count[i] = 0
            ep_oscillation[i] = 0
            prev_xy_err[i] = float('inf')
            prev_xy_decreasing[i] = False

        obs, _ = env_wrapped.reset()

    # Progress
    if step % 1000 == 0:
        print(f"  step {step:>5}  xy_err={xy_err.mean():.3f}  z_off={z_offset.mean():.3f}  "
              f"vz={vz.mean():.3f}  tilt={tilt_deg.mean():.1f}  ov={overlap_ratio.mean():.3f}")

# === Results ===
import statistics as _stats

def _s(vals, fmt=".3f"):
    if not vals: return "(n=0)"
    m = sum(vals)/len(vals)
    md = _stats.median(vals)
    return f"mean={m:{fmt}} med={md:{fmt}} min={min(vals):{fmt}} max={max(vals):{fmt}} n={len(vals)}"

n_ep = max(results["total_episodes"], 1)

print(f"\n{'='*70}")
print(f"  ANALYTICAL BASE CONTROLLER — PD GAIN TUNING RESULTS")
print(f"{'='*70}")
print(f"  Total episodes:     {results['total_episodes']}")
print(f"  Dock success:       {results['dock_success']}/{n_ep} ({100*results['dock_success']/n_ep:.1f}%)")
print(f"  Crashed:            {results['crashed']}/{n_ep} ({100*results['crashed']/n_ep:.0f}%)")
print(f"  Reached <10cm:      {results['reached_10cm']}/{n_ep} ({100*results['reached_10cm']/n_ep:.0f}%)")
print(f"  Reached <5cm:       {results['reached_5cm']}/{n_ep} ({100*results['reached_5cm']/n_ep:.0f}%)")

print(f"\n  --- XY Convergence ---")
print(f"  Min XY err:    {_s(results['min_xy_err'])}")
print(f"  Min XY step:   {_s(results['min_xy_step'], '.0f')}")
print(f"  XY oscillation: {_s(results['xy_oscillation_count'], '.0f')}")
high_osc = sum(1 for o in results['xy_oscillation_count'] if o > 20)
print(f"  High oscillation (>20 changes): {high_osc}/{n_ep} ({100*high_osc/n_ep:.0f}%)")

print(f"\n  --- Descent / Z Performance ---")
print(f"  End z_offset:  {_s(results['end_z_offset'])}")
above = sum(1 for z in results['end_z_offset'] if z > 0.02)
below = sum(1 for z in results['end_z_offset'] if z < -0.02)
near = sum(1 for z in results['end_z_offset'] if -0.02 <= z <= 0.02)
print(f"  End position:  above box: {above} ({100*above/n_ep:.0f}%)  "
      f"near: {near} ({100*near/n_ep:.0f}%)  below: {below} ({100*below/n_ep:.0f}%)")
print(f"  End v_z:       {_s(results['end_vz'])}")
print(f"  End tilt:      {_s(results['end_tilt_deg'], '.1f')}")

print(f"\n  --- Overlap / Dock ---")
print(f"  Episodes with overlap>0: {len(results['first_overlap_step'])}/{n_ep} "
      f"({100*len(results['first_overlap_step'])/n_ep:.0f}%)")
if results['first_overlap_step']:
    print(f"  First overlap step: {_s(results['first_overlap_step'], '.0f')}")

print(f"\n  --- XY Convergence Timeline (means across envs) ---")
print(f"  {'step':>6}  {'xy_err':>8}  {'z_offset':>9}  {'v_z':>7}  {'tilt':>6}  {'overlap':>8}")
for i in range(0, len(xy_err_time), max(1, len(xy_err_time)//15)):
    s, xy = xy_err_time[i]
    _, zo = z_offset_time[i]
    _, vz_v = vz_time[i]
    _, ti = tilt_time[i]
    _, ov = overlap_time[i]
    print(f"  {s:>6}  {xy:>8.3f}  {zo:>9.3f}  {vz_v:>+7.3f}  {ti:>6.1f}  {ov:>8.4f}")

# PD gain diagnosis
print(f"\n  --- PD GAIN DIAGNOSIS ---")
osc_mean = sum(results['xy_oscillation_count']) / n_ep if results['xy_oscillation_count'] else 0
if osc_mean > 15:
    print(f"  ⚠️  XY oscillation high ({osc_mean:.0f}): D gain too low. Increase D from 3.0 → 4.5")
elif osc_mean < 5:
    print(f"  ⚠️  XY oscillation very low ({osc_mean:.0f}): D gain may be too high (overdamped)")
else:
    print(f"  ✅  XY oscillation OK ({osc_mean:.0f}): P/D balance reasonable")

above_pct = 100 * above / n_ep
if above_pct > 70:
    print(f"  ⚠️  {above_pct:.0f}% end above box: descent too slow. Increase desired_vz or widen gate")
elif above_pct < 30:
    print(f"  ✅  Descent adequate ({above_pct:.0f}% above at end)")
else:
    print(f"  🟡  {above_pct:.0f}% above at end: descent marginal")

crash_pct = 100 * results['crashed'] / n_ep
if crash_pct > 15:
    print(f"  ⚠️  Crash rate {crash_pct:.0f}%: reduce P gain or increase safety push")
else:
    print(f"  ✅  Crash rate OK ({crash_pct:.0f}%)")

reach_5 = 100 * results['reached_5cm'] / n_ep
if reach_5 < 30:
    print(f"  ⚠️  Only {reach_5:.0f}% reach <5cm: P gain may be too low, or D too high")
else:
    print(f"  ✅  {reach_5:.0f}% reach <5cm: XY tracking adequate")

print(f"{'='*70}")

env.close()
simulation_app.close()
