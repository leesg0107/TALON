"""Stage 3 evaluation: approach + containment + grasp."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import time
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 100
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 8.0
cfg.lock_gripper = True
# Dynamic box
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

# --- SELECT MODEL ---
CKPT = "logs/stage3_safety_v1/best_agent.pt"

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path=CKPT)
agent.set_running_mode("eval")

# --- Capture Column Visualization ---
# Column: X=±0.05m, Y=±0.104m, extends from gripper down to ground
column_marker_cfg = VisualizationMarkersCfg(
    prim_path="/Visuals/CaptureColumn",
    markers={
        "column": sim_utils.CuboidCfg(
            size=(0.10, 0.208, 0.50),  # X=10cm, Y=20.8cm, Z=50cm tall
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                opacity=0.15,
            ),
        ),
    },
)
column_markers = VisualizationMarkers(column_marker_cfg)

# --- Metrics ---
num_episodes = 0
num_envs = env.num_envs
episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

# Per-env tracking
align_count = torch.zeros(num_envs, dtype=torch.long, device=device)
gripper_closed = torch.zeros(num_envs, dtype=torch.bool, device=device)
docked = torch.zeros(num_envs, dtype=torch.bool, device=device)  # overlap>90% achieved
dock_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)  # step when docked
grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)  # box lifted after grip
grasp_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

# Aggregated results
results = {
    "dock_success": 0,      # overlap>90% 달성 횟수
    "dock_time_sum": 0.0,   # 안착까지 시간 합산
    "grasp_success": 0,     # 파지 성공 (box lift) 횟수
    "grasp_time_sum": 0.0,  # 파지까지 시간 합산
    "hold_steps": 0,        # 파지 유지 총 steps
    "total_episodes": 0,
    "xy_err_at_dock": [],   # 안착 시 XY 오프셋
}

# === Per-episode diagnostic state (reset on episode end) ===
ep_min_xy_err = torch.full((num_envs,), float('inf'), device=device)
ep_min_xy_err_z_offset = torch.zeros(num_envs, device=device)
ep_min_xy_err_above = torch.zeros(num_envs, dtype=torch.bool, device=device)
# Local-frame x/y at moment of min XY (for X-precision hypothesis test)
ep_min_xy_x_local = torch.zeros(num_envs, device=device)
ep_min_xy_y_local = torch.zeros(num_envs, device=device)
# Per-episode min |x_local| and min |y_local| independently
ep_min_x_local = torch.full((num_envs,), float('inf'), device=device)
ep_min_y_local = torch.full((num_envs,), float('inf'), device=device)
ep_first_overlap_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
# State snapshots at first overlap > 0 moment
ep_first_overlap_x_local = torch.zeros(num_envs, device=device)
ep_first_overlap_y_local = torch.zeros(num_envs, device=device)
ep_first_overlap_tilt = torch.zeros(num_envs, device=device)
ep_first_overlap_vz = torch.zeros(num_envs, device=device)
ep_first_overlap_z_offset = torch.zeros(num_envs, device=device)
ep_bounce_count = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_prev_overlap_pos = torch.zeros(num_envs, dtype=torch.bool, device=device)

# Rolling history (last HIST_LEN steps) for pre-failure capture
HIST_LEN = 3
hist_vz = torch.zeros(num_envs, HIST_LEN, device=device)
hist_vxy = torch.zeros(num_envs, HIST_LEN, device=device)
hist_tilt = torch.zeros(num_envs, HIST_LEN, device=device)
hist_xy_err = torch.zeros(num_envs, HIST_LEN, device=device)
hist_gripper_z = torch.zeros(num_envs, HIST_LEN, device=device)
hist_idx = torch.zeros(num_envs, dtype=torch.long, device=device)  # circular pointer

# Diagnostic accumulators (lists)
diag = {
    # Failure-only metrics (captured at last step before reset)
    "fail_tilt_at_end_deg": [],
    "fail_vz_at_end": [],
    "fail_vxy_at_end": [],
    "fail_xy_err_at_end": [],
    "fail_x_local_at_end": [],       # |box_local_x| at last step (for X-precision test)
    "fail_y_local_at_end": [],       # |box_local_y|
    "fail_approach_angle_deg": [],   # 90=pure vertical descent, 0=horizontal swipe
    "fail_z_offset_at_end": [],      # gripper_z - box_top (negative = below top)
    # All-episode metrics
    "min_xy_err": [],
    "min_xy_err_z_offset": [],       # gripper_z - box_top at min xy moment
    "min_xy_err_above_box": [],      # bool: was gripper above box top at min xy?
    "min_xy_x_local": [],            # |x_local| at moment of min xy_err
    "min_xy_y_local": [],            # |y_local| at moment of min xy_err
    "min_x_local_indep": [],         # per-episode min |x_local| (independent of min xy)
    "min_y_local_indep": [],
    "first_overlap_step": [],        # only for episodes with any overlap (-1 excluded)
    # Snapshots at first overlap > 0 moment
    "first_overlap_x_local": [],
    "first_overlap_y_local": [],
    "first_overlap_tilt_deg": [],
    "first_overlap_vz": [],
    "first_overlap_z_offset": [],
    "bounce_count": [],
    # Dock-success-only metrics
    "dock_x_local": [],
    "dock_y_local": [],
    "dock_bounce_count": [],         # bounce count at moment of dock success
    "dock_tilt_deg": [],             # tilt at dock
    "dock_vz": [],                   # v_z at dock
    "dock_z_offset": [],             # gripper_z - box_top at dock
    "dock_first_overlap_step": [],   # first overlap step (how early overlap started)
    "dock_step": [],                 # step when dock triggered
    # Success: state at FIRST overlap moment (for success vs failure comparison)
    "dock_first_ov_x_local": [],
    "dock_first_ov_y_local": [],
    "dock_first_ov_tilt_deg": [],
    "dock_first_ov_vz": [],
    "dock_first_ov_z_offset": [],
    "dock_min_xy_err": [],           # best XY precision achieved during episode
    # Pre-failure 3-step history (failure episodes only)
    "pre_fail_vz_seq": [],
    "pre_fail_tilt_deg_seq": [],
    "pre_fail_xy_err_seq": [],
}

max_eval_steps = 12000  # 2 episodes per env → ~200 episodes total
print(f"\n=== Evaluation: {num_envs} envs x {max_eval_steps} steps ===\n")

obs, _ = env_wrapped.reset()
env_step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

for step in range(max_eval_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_eval_steps)[0]

    # --- Compute overlap in gripper local frame ---
    pos_w = env.robot.data.root_pos_w
    quat_w = env.robot.data.root_quat_w
    from envs.drone_env import quat_to_rot_matrix
    R = quat_to_rot_matrix(quat_w)
    gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=device)
    gripper_pos_w = pos_w + torch.bmm(R, gripper_offset_b.expand(num_envs, -1).unsqueeze(-1)).squeeze(-1)

    box_offset_w = env.object_pos - gripper_pos_w
    box_local = torch.bmm(R.transpose(1, 2), box_offset_w.unsqueeze(-1)).squeeze(-1)

    box_half = 0.04
    gripper_half_x, gripper_half_y = 0.05, 0.062

    box_min_x = box_local[:, 0] - box_half
    box_max_x = box_local[:, 0] + box_half
    box_min_y = box_local[:, 1] - box_half
    box_max_y = box_local[:, 1] + box_half

    overlap_x = (torch.min(torch.full_like(box_max_x, gripper_half_x), box_max_x)
               - torch.max(torch.full_like(box_min_x, -gripper_half_x), box_min_x)).clamp(min=0.0)
    overlap_y = (torch.min(torch.full_like(box_max_y, gripper_half_y), box_max_y)
               - torch.max(torch.full_like(box_min_y, -gripper_half_y), box_min_y)).clamp(min=0.0)

    box_area = (2 * box_half) ** 2
    overlap_xy = (overlap_x * overlap_y) / box_area
    z_in_range = ((box_local[:, 2] > -0.12) & (box_local[:, 2] < 0.02)).float()
    overlap_ratio = overlap_xy * z_in_range

    xy_err = torch.norm(gripper_pos_w[:, :2] - env.object_pos[:, :2], dim=-1)

    # --- Diagnostic: record per-step state ---
    vel_w = env.robot.data.root_lin_vel_w  # (N, 3)
    vel_xy_mag = torch.norm(vel_w[:, :2], dim=-1)
    vel_z = vel_w[:, 2]
    tilt_angle = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))  # rad
    tilt_deg_now = tilt_angle * (180.0 / 3.14159)
    box_top = env.object_pos[:, 2] + 0.04  # box center + half (8cm box)
    z_offset = gripper_pos_w[:, 2] - box_top

    # Local-frame x/y errors (for X-precision hypothesis test)
    x_local_abs = torch.abs(box_local[:, 0])
    y_local_abs = torch.abs(box_local[:, 1])

    # Track min xy_err per episode and state at that moment
    is_new_min = xy_err < ep_min_xy_err
    ep_min_xy_err = torch.where(is_new_min, xy_err, ep_min_xy_err)
    ep_min_xy_err_z_offset = torch.where(is_new_min, z_offset, ep_min_xy_err_z_offset)
    ep_min_xy_err_above = torch.where(is_new_min, z_offset > 0, ep_min_xy_err_above)
    ep_min_xy_x_local = torch.where(is_new_min, x_local_abs, ep_min_xy_x_local)
    ep_min_xy_y_local = torch.where(is_new_min, y_local_abs, ep_min_xy_y_local)

    # Independent per-axis min
    ep_min_x_local = torch.minimum(ep_min_x_local, x_local_abs)
    ep_min_y_local = torch.minimum(ep_min_y_local, y_local_abs)

    # First overlap step + snapshot at that moment
    overlap_pos_now = overlap_ratio > 0.0
    first_overlap_mask = overlap_pos_now & (ep_first_overlap_step < 0)
    ep_first_overlap_step = torch.where(
        first_overlap_mask, env_step_count, ep_first_overlap_step
    )
    ep_first_overlap_x_local = torch.where(first_overlap_mask, x_local_abs, ep_first_overlap_x_local)
    ep_first_overlap_y_local = torch.where(first_overlap_mask, y_local_abs, ep_first_overlap_y_local)
    ep_first_overlap_tilt = torch.where(first_overlap_mask, tilt_deg_now, ep_first_overlap_tilt)
    ep_first_overlap_vz = torch.where(first_overlap_mask, vel_z, ep_first_overlap_vz)
    ep_first_overlap_z_offset = torch.where(first_overlap_mask, z_offset, ep_first_overlap_z_offset)

    # Bounce count: overlap>0 → overlap=0 transitions
    bounce_mask = ep_prev_overlap_pos & (~overlap_pos_now)
    ep_bounce_count += bounce_mask.long()
    ep_prev_overlap_pos = overlap_pos_now

    # Rolling history write (circular buffer)
    arange_n = torch.arange(num_envs, device=device)
    hist_vz[arange_n, hist_idx] = vel_z
    hist_vxy[arange_n, hist_idx] = vel_xy_mag
    hist_tilt[arange_n, hist_idx] = tilt_deg_now
    hist_xy_err[arange_n, hist_idx] = xy_err
    hist_gripper_z[arange_n, hist_idx] = gripper_pos_w[:, 2]
    hist_idx = (hist_idx + 1) % HIST_LEN

    # --- Gripper control: open by default, close when docked ---
    for i in range(num_envs):
        ov = overlap_ratio[i].item()

        if gripper_closed[i]:
            action[i, 7] = -1.0  # keep closed
        else:
            action[i, 7] = 1.0  # keep open

            # Check for docking (overlap > 50% cumulative, same as training)
            if ov > 0.50:
                align_count[i] += 1
            # No reset: cumulative

            # Docked after cumulative ~1s (150 steps, same as training)
            if align_count[i] > 150:
                action[i, 7] = -1.0
                gripper_closed[i] = True
                if not docked[i]:
                    docked[i] = True
                    dock_step[i] = env_step_count[i].clone()
                    results["dock_success"] += 1
                    dt = env_step_count[i].item() / 150.0
                    results["dock_time_sum"] += dt
                    results["xy_err_at_dock"].append(xy_err[i].item())
                    # Dock success snapshot
                    diag["dock_x_local"].append(abs(box_local[i, 0].item()))
                    diag["dock_y_local"].append(abs(box_local[i, 1].item()))
                    diag["dock_bounce_count"].append(int(ep_bounce_count[i].item()))
                    diag["dock_tilt_deg"].append(tilt_deg_now[i].item())
                    diag["dock_vz"].append(vel_z[i].item())
                    diag["dock_z_offset"].append(z_offset[i].item())
                    diag["dock_first_overlap_step"].append(int(ep_first_overlap_step[i].item()))
                    diag["dock_step"].append(int(env_step_count[i].item()))
                    diag["dock_first_ov_x_local"].append(ep_first_overlap_x_local[i].item())
                    diag["dock_first_ov_y_local"].append(ep_first_overlap_y_local[i].item())
                    diag["dock_first_ov_tilt_deg"].append(ep_first_overlap_tilt[i].item())
                    diag["dock_first_ov_vz"].append(ep_first_overlap_vz[i].item())
                    diag["dock_first_ov_z_offset"].append(ep_first_overlap_z_offset[i].item())
                    diag["dock_min_xy_err"].append(ep_min_xy_err[i].item())

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    env_step_count += 1

    # Update capture column visualization (skip in headless)
    # if step % 5 == 0:
    #     col_pos = gripper_pos_w.clone()
    #     col_pos[:, 2] -= 0.25
    #     col_quat = quat_w.clone()
    #     column_markers.visualize(col_pos, col_quat)

    # --- Check grasp: box lifted after gripper closed ---
    for i in range(num_envs):
        if gripper_closed[i] and not grasped[i]:
            box_z = env.object_pos[i, 2].item()
            if box_z > 0.55:  # box lifted above rest height (0.5m) + 5cm
                grasped[i] = True
                grasp_step[i] = env_step_count[i].clone()
                results["grasp_success"] += 1
                dt = env_step_count[i].item() / 150.0
                results["grasp_time_sum"] += dt

        if grasped[i]:
            results["hold_steps"] += 1

    # --- Print status ---
    if step % 300 == 0:
        status = ""
        for i in range(min(num_envs, 4)):
            ov = overlap_ratio[i].item()
            xy = xy_err[i].item()
            d = "D" if docked[i] else "."
            g = "G" if grasped[i] else "."
            c = "C" if gripper_closed[i] else "O"
            status += f"  env{i}:ov={ov*100:.0f}% xy={xy:.2f} {d}{g}{c}"
        print(f"  step {step:4d}{status}")

    # --- Episode reset + failure classification ---
    # Flatten in case shape is (N,1)
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        reset_list = done_mask.nonzero(as_tuple=False).view(-1).tolist()
        results["total_episodes"] += len(reset_list)
        for i in reset_list:
            is_docked = docked[i].item()

            # --- Diagnostic snapshots (per episode) ---
            diag["min_xy_err"].append(ep_min_xy_err[i].item())
            diag["min_xy_err_z_offset"].append(ep_min_xy_err_z_offset[i].item())
            diag["min_xy_err_above_box"].append(bool(ep_min_xy_err_above[i].item()))
            diag["min_xy_x_local"].append(ep_min_xy_x_local[i].item())
            diag["min_xy_y_local"].append(ep_min_xy_y_local[i].item())
            min_x_v = ep_min_x_local[i].item()
            min_y_v = ep_min_y_local[i].item()
            if min_x_v != float('inf'):
                diag["min_x_local_indep"].append(min_x_v)
            if min_y_v != float('inf'):
                diag["min_y_local_indep"].append(min_y_v)
            if ep_first_overlap_step[i].item() >= 0:
                diag["first_overlap_step"].append(int(ep_first_overlap_step[i].item()))
                diag["first_overlap_x_local"].append(ep_first_overlap_x_local[i].item())
                diag["first_overlap_y_local"].append(ep_first_overlap_y_local[i].item())
                diag["first_overlap_tilt_deg"].append(ep_first_overlap_tilt[i].item())
                diag["first_overlap_vz"].append(ep_first_overlap_vz[i].item())
                diag["first_overlap_z_offset"].append(ep_first_overlap_z_offset[i].item())
            diag["bounce_count"].append(int(ep_bounce_count[i].item()))

            if not is_docked:
                drone_z = pos_w[i, 2].item()
                box_z = env.object_pos[i, 2].item()
                xy = xy_err[i].item()
                is_term = term_flat[i].item()

                # --- Failure-only diagnostics: state at last step ---
                diag["fail_tilt_at_end_deg"].append(tilt_deg_now[i].item())
                diag["fail_vz_at_end"].append(vel_z[i].item())
                diag["fail_vxy_at_end"].append(vel_xy_mag[i].item())
                diag["fail_xy_err_at_end"].append(xy)
                diag["fail_z_offset_at_end"].append(z_offset[i].item())
                diag["fail_x_local_at_end"].append(x_local_abs[i].item())
                diag["fail_y_local_at_end"].append(y_local_abs[i].item())
                # Approach angle: 90 = pure vertical descent, 0 = horizontal swipe
                # Only meaningful when descending; clamp vel_z denominator
                vz_i = vel_z[i].item()
                vxy_i = vel_xy_mag[i].item()
                import math as _math
                if vz_i < -0.01:
                    angle_deg = _math.degrees(_math.atan2(-vz_i, max(vxy_i, 1e-3)))
                else:
                    angle_deg = 0.0  # not descending → not a "contact angle"
                diag["fail_approach_angle_deg"].append(angle_deg)

                # Pre-fail 3-step history (chronological order)
                cur_idx = int(hist_idx[i].item())  # next write position
                ordered = [(cur_idx + k) % HIST_LEN for k in range(HIST_LEN)]
                diag["pre_fail_vz_seq"].append([hist_vz[i, k].item() for k in ordered])
                diag["pre_fail_tilt_deg_seq"].append([hist_tilt[i, k].item() for k in ordered])
                diag["pre_fail_xy_err_seq"].append([hist_xy_err[i, k].item() for k in ordered])

                if is_term and drone_z < 0.15:
                    results.setdefault("fail_crash", 0)
                    results["fail_crash"] += 1
                elif xy > 0.5:
                    results.setdefault("fail_far", 0)
                    results["fail_far"] += 1
                elif xy <= 0.5 and drone_z < box_z:
                    results.setdefault("fail_below", 0)
                    results["fail_below"] += 1
                elif xy <= 0.5 and drone_z >= box_z:
                    results.setdefault("fail_near_timeout", 0)
                    results["fail_near_timeout"] += 1
                else:
                    results.setdefault("fail_other", 0)
                    results["fail_other"] += 1

            # Reset per-env state
            align_count[i] = 0
            gripper_closed[i] = False
            docked[i] = False
            dock_step[i] = -1
            grasped[i] = False
            grasp_step[i] = -1
            env_step_count[i] = 0
            ep_min_xy_err[i] = float('inf')
            ep_min_xy_err_z_offset[i] = 0.0
            ep_min_xy_err_above[i] = False
            ep_min_xy_x_local[i] = 0.0
            ep_min_xy_y_local[i] = 0.0
            ep_min_x_local[i] = float('inf')
            ep_min_y_local[i] = float('inf')
            ep_first_overlap_step[i] = -1
            ep_first_overlap_x_local[i] = 0.0
            ep_first_overlap_y_local[i] = 0.0
            ep_first_overlap_tilt[i] = 0.0
            ep_first_overlap_vz[i] = 0.0
            ep_first_overlap_z_offset[i] = 0.0
            ep_bounce_count[i] = 0
            ep_prev_overlap_pos[i] = False
            hist_vz[i].zero_()
            hist_vxy[i].zero_()
            hist_tilt[i].zero_()
            hist_xy_err[i].zero_()
            hist_gripper_z[i].zero_()
            hist_idx[i] = 0
        # Isaac Lab auto-resets terminated/truncated envs inside step()
        # Do NOT call env_wrapped.reset() — it resets ALL envs

# --- Final Results ---
total_ep = max(results["total_episodes"], 1)
dock_succ = results["dock_success"]
grasp_succ = results["grasp_success"]

print(f"\n{'='*60}")
print(f"  EVALUATION RESULTS")
print(f"{'='*60}")
print(f"  Total episodes:     {results['total_episodes']}")
print(f"  Dock success:       {dock_succ}/{results['total_episodes']} ({100*dock_succ/total_ep:.1f}%)")
if dock_succ > 0:
    print(f"  Avg dock time:      {results['dock_time_sum']/dock_succ:.1f}s")
    avg_xy = sum(results['xy_err_at_dock']) / len(results['xy_err_at_dock'])
    print(f"  Avg XY err at dock: {avg_xy*100:.1f}cm")
print(f"  Grasp success:      {grasp_succ}/{results['total_episodes']} ({100*grasp_succ/total_ep:.1f}%)")
if grasp_succ > 0:
    print(f"  Avg grasp time:     {results['grasp_time_sum']/grasp_succ:.1f}s")
print(f"  Hold steps:         {results['hold_steps']}")
print(f"\n  --- Failure Analysis ---")
fail_total = total_ep - dock_succ
print(f"  Total failures:     {fail_total}")
print(f"  Crash (hit ground): {results.get('fail_crash', 0)} ({100*results.get('fail_crash',0)/max(fail_total,1):.0f}%)")
print(f"  Far (xy>0.5m):      {results.get('fail_far', 0)} ({100*results.get('fail_far',0)/max(fail_total,1):.0f}%)")
print(f"  Below box:          {results.get('fail_below', 0)} ({100*results.get('fail_below',0)/max(fail_total,1):.0f}%)")
print(f"  Near but timeout:   {results.get('fail_near_timeout', 0)} ({100*results.get('fail_near_timeout',0)/max(fail_total,1):.0f}%)")
print(f"  Other:              {results.get('fail_other', 0)}")
print(f"{'='*60}")

# === Crash diagnostic summary ===
import statistics as _stats

def _summary(name, vals, fmt="{:.3f}"):
    if not vals:
        print(f"  {name:<32} (n=0)")
        return
    mn = min(vals); mx = max(vals); mean = sum(vals) / len(vals)
    med = _stats.median(vals)
    print(f"  {name:<32} n={len(vals):<4} mean={fmt.format(mean)} med={fmt.format(med)} "
          f"min={fmt.format(mn)} max={fmt.format(mx)}")

def _percentile_buckets(name, vals, edges, unit=""):
    if not vals:
        print(f"  {name}: (n=0)"); return
    n = len(vals)
    print(f"  {name} (n={n}):")
    lo_label = "-inf"
    prev = float('-inf')
    for e in edges:
        cnt = sum(1 for v in vals if prev <= v < e)
        print(f"    [{lo_label:>6}, {e:>6}){unit}: {cnt:>4} ({100*cnt/n:.0f}%)")
        prev = e
        lo_label = str(e)
    cnt = sum(1 for v in vals if v >= prev)
    print(f"    [{lo_label:>6},   +inf){unit}: {cnt:>4} ({100*cnt/n:.0f}%)")

print(f"\n{'='*60}")
print(f"  CRASH / FAILURE DIAGNOSTICS")
print(f"{'='*60}")

print(f"\n  --- All-episode metrics ---")
_summary("Min XY err per episode [m]", diag["min_xy_err"])
_summary("Z offset at min XY [m]",      diag["min_xy_err_z_offset"])
above_count = sum(1 for b in diag["min_xy_err_above_box"] if b)
n_ep = max(len(diag["min_xy_err_above_box"]), 1)
print(f"  Min XY achieved ABOVE box top:  {above_count}/{n_ep} ({100*above_count/n_ep:.0f}%)")
_summary("First overlap step (only ep w/ overlap)", diag["first_overlap_step"], "{:.0f}")
print(f"    (episodes that ever achieved overlap>0: {len(diag['first_overlap_step'])}/{n_ep})")
_summary("Bounce count (overlap+ → overlap0)", diag["bounce_count"], "{:.1f}")
multi_bounce = sum(1 for b in diag["bounce_count"] if b >= 2)
print(f"  Episodes with >=2 bounces:      {multi_bounce}/{n_ep} ({100*multi_bounce/n_ep:.0f}%)")

print(f"\n  --- Failure-only: state at last step ---")
_summary("Tilt at end [deg]",    diag["fail_tilt_at_end_deg"], "{:.1f}")
_summary("v_z at end [m/s]",     diag["fail_vz_at_end"])
_summary("v_xy at end [m/s]",    diag["fail_vxy_at_end"])
_summary("XY err at end [m]",    diag["fail_xy_err_at_end"])
_summary("|x_local| at end [m]", diag["fail_x_local_at_end"])
_summary("|y_local| at end [m]", diag["fail_y_local_at_end"])
_summary("Z offset at end [m]",  diag["fail_z_offset_at_end"])
_summary("Approach angle [deg] (90=vert)", diag["fail_approach_angle_deg"], "{:.1f}")

# === X-precision hypothesis test ===
print(f"\n  ===== X-PRECISION HYPOTHESIS TEST =====")
print(f"  Gripper clearances: X=1.0cm, Y=2.2cm")

def _std(vals):
    if len(vals) < 2: return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

# Distribution at moment of min XY (best alignment per episode)
print(f"\n  --- At moment of MIN XY err (per episode) ---")
_summary("|x_local| @ min_xy [cm]",
         [v*100 for v in diag["min_xy_x_local"]], "{:.2f}")
_summary("|y_local| @ min_xy [cm]",
         [v*100 for v in diag["min_xy_y_local"]], "{:.2f}")
if diag["min_xy_x_local"]:
    sx = _std(diag["min_xy_x_local"]) * 100
    sy = _std(diag["min_xy_y_local"]) * 100
    print(f"  std(|x_local|) = {sx:.2f}cm   std(|y_local|) = {sy:.2f}cm")
    print(f"  ratio sx/sy = {sx/max(sy,1e-6):.2f}  (>>1 supports X-bottleneck)")

# Independent per-axis min (best each axis ever achieved)
_summary("min |x_local| ever [cm]",
         [v*100 for v in diag["min_x_local_indep"]], "{:.2f}")
_summary("min |y_local| ever [cm]",
         [v*100 for v in diag["min_y_local_indep"]], "{:.2f}")

# 4-category failure breakdown using x/y at end (failure-only)
print(f"\n  --- Failure category by clearance (n={len(diag['fail_x_local_at_end'])}) ---")
xs = diag["fail_x_local_at_end"]
ys = diag["fail_y_local_at_end"]
X_TH, Y_TH = 0.010, 0.022
nf = max(len(xs), 1)
cat1 = sum(1 for x, y in zip(xs, ys) if x < X_TH and y < Y_TH)
cat2 = sum(1 for x, y in zip(xs, ys) if x >= X_TH and y < Y_TH)
cat3 = sum(1 for x, y in zip(xs, ys) if x < X_TH and y >= Y_TH)
cat4 = sum(1 for x, y in zip(xs, ys) if x >= X_TH and y >= Y_TH)
print(f"  [|x|<1cm AND |y|<2.2cm] both OK but failed: {cat1:>4} ({100*cat1/nf:.0f}%)")
print(f"  [|x|>1cm AND |y|<2.2cm] PURE X failure:     {cat2:>4} ({100*cat2/nf:.0f}%)  <-- supports hypothesis")
print(f"  [|x|<1cm AND |y|>2.2cm] PURE Y failure:     {cat3:>4} ({100*cat3/nf:.0f}%)")
print(f"  [|x|>1cm AND |y|>2.2cm] both fail (far):    {cat4:>4} ({100*cat4/nf:.0f}%)")

# Distribution of |x_local| at failure end
_percentile_buckets("|x_local| at end [m]", xs,
                    [0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100], "m")
_percentile_buckets("|y_local| at end [m]", ys,
                    [0.005, 0.010, 0.020, 0.030, 0.050, 0.100], "m")

# === First-overlap snapshot ===
print(f"\n  --- At FIRST overlap > 0 moment (n={len(diag['first_overlap_x_local'])}) ---")
_summary("|x_local| @ first_overlap [cm]",
         [v*100 for v in diag["first_overlap_x_local"]], "{:.2f}")
_summary("|y_local| @ first_overlap [cm]",
         [v*100 for v in diag["first_overlap_y_local"]], "{:.2f}")
_summary("tilt @ first_overlap [deg]",
         diag["first_overlap_tilt_deg"], "{:.1f}")
_summary("v_z @ first_overlap [m/s]",
         diag["first_overlap_vz"], "{:.3f}")
_summary("z_offset @ first_overlap [cm]",
         [v*100 for v in diag["first_overlap_z_offset"]], "{:.2f}")

# === Dock success snapshot ===
print(f"\n  --- At DOCK success moment (n={len(diag['dock_x_local'])}) ---")
if diag["dock_x_local"]:
    _summary("|x_local| @ dock [cm]",
             [v*100 for v in diag["dock_x_local"]], "{:.2f}")
    _summary("|y_local| @ dock [cm]",
             [v*100 for v in diag["dock_y_local"]], "{:.2f}")
    sx_dock = _std(diag["dock_x_local"]) * 100
    sy_dock = _std(diag["dock_y_local"]) * 100
    print(f"  std(|x_local|@dock) = {sx_dock:.2f}cm   std(|y_local|@dock) = {sy_dock:.2f}cm")
    _summary("Bounce count @ dock", diag["dock_bounce_count"], "{:.1f}")
    zero_bounce = sum(1 for b in diag["dock_bounce_count"] if b == 0)
    n_dock = len(diag["dock_bounce_count"])
    print(f"  Zero-bounce docks (clean):     {zero_bounce}/{n_dock} ({100*zero_bounce/max(n_dock,1):.0f}%)")
    multi_bounce = sum(1 for b in diag["dock_bounce_count"] if b >= 2)
    print(f"  Multi-bounce docks (>=2):      {multi_bounce}/{n_dock} ({100*multi_bounce/max(n_dock,1):.0f}%)")
    _summary("Tilt @ dock [deg]", diag["dock_tilt_deg"], "{:.1f}")
    _summary("v_z @ dock [m/s]", diag["dock_vz"], "{:.3f}")
    _summary("Z offset @ dock [cm]", [v*100 for v in diag["dock_z_offset"]], "{:.1f}")
    _summary("First overlap step (dock eps)", diag["dock_first_overlap_step"], "{:.0f}")
    _summary("Dock step", diag["dock_step"], "{:.0f}")
    # Dynamic readiness: zero-bounce + low tilt = clean vertical dock
    clean_dock = sum(1 for b, t in zip(diag["dock_bounce_count"], diag["dock_tilt_deg"]) if b == 0 and t < 10)
    print(f"\n  ** Dynamic readiness: clean dock (0 bounce + tilt<10°): "
          f"{clean_dock}/{n_dock} ({100*clean_dock/max(n_dock,1):.0f}%) **")

    # Success: first overlap snapshot
    print(f"\n  --- Success: state at FIRST overlap ---")
    _summary("|x_local| @ 1st_ov [cm]",
             [v*100 for v in diag["dock_first_ov_x_local"]], "{:.2f}")
    _summary("|y_local| @ 1st_ov [cm]",
             [v*100 for v in diag["dock_first_ov_y_local"]], "{:.2f}")
    _summary("tilt @ 1st_ov [deg]", diag["dock_first_ov_tilt_deg"], "{:.1f}")
    _summary("v_z @ 1st_ov [m/s]", diag["dock_first_ov_vz"], "{:.3f}")
    _summary("z_offset @ 1st_ov [cm]",
             [v*100 for v in diag["dock_first_ov_z_offset"]], "{:.1f}")
    _summary("min_xy_err [cm]",
             [v*100 for v in diag["dock_min_xy_err"]], "{:.2f}")

# === SUCCESS vs FAILURE comparison ===
# Separate first_overlap data by outcome
fail_first_ov_vz = []
fail_first_ov_tilt = []
fail_first_ov_x = []
fail_first_ov_y = []
fail_first_ov_z_off = []
fail_min_xy = []
# Reconstruct from all-episode data minus dock success data
# All episodes with overlap: first_overlap_* lists
# Dock episodes: dock_first_ov_* lists
# Failure with overlap = all_overlap - dock
n_all_ov = len(diag["first_overlap_vz"])
n_dock_ov = len(diag["dock_first_ov_vz"])
if n_all_ov > n_dock_ov:
    # We need to reconstruct failure-only first_overlap data
    # Since we appended in order, dock entries are interleaved
    # Instead, compute from the full lists and dock lists by difference of means
    pass

print(f"\n{'='*60}")
print(f"  SUCCESS vs FAILURE: first overlap comparison")
print(f"{'='*60}")
if diag["dock_first_ov_vz"] and diag["first_overlap_vz"]:
    # All-overlap = success + failure-with-overlap
    all_vz = diag["first_overlap_vz"]
    dock_vz = diag["dock_first_ov_vz"]
    all_tilt = diag["first_overlap_tilt_deg"]
    dock_tilt = diag["dock_first_ov_tilt_deg"]
    all_x = diag["first_overlap_x_local"]
    dock_x = diag["dock_first_ov_x_local"]

    n_all = len(all_vz)
    n_dock = len(dock_vz)
    n_fail_ov = n_all - n_dock

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0

    def _med(vals):
        if not vals: return 0
        s = sorted(vals)
        return s[len(s)//2]

    print(f"  Episodes with overlap: {n_all} (success: {n_dock}, failure: {n_fail_ov})")
    print(f"")
    print(f"  {'Metric':<25} {'Success':>12} {'All w/overlap':>14} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*10}")

    m_dock_vz = _mean(dock_vz)
    m_all_vz = _mean(all_vz)
    print(f"  {'v_z mean [m/s]':<25} {m_dock_vz:>12.3f} {m_all_vz:>14.3f} {m_dock_vz-m_all_vz:>+10.3f}")

    m_dock_vz_med = _med(dock_vz)
    m_all_vz_med = _med(all_vz)
    print(f"  {'v_z median [m/s]':<25} {m_dock_vz_med:>12.3f} {m_all_vz_med:>14.3f} {m_dock_vz_med-m_all_vz_med:>+10.3f}")

    m_dock_tilt = _mean(dock_tilt)
    m_all_tilt = _mean(all_tilt)
    print(f"  {'tilt mean [deg]':<25} {m_dock_tilt:>12.1f} {m_all_tilt:>14.1f} {m_dock_tilt-m_all_tilt:>+10.1f}")

    m_dock_x = _mean([v*100 for v in dock_x])
    m_all_x = _mean([v*100 for v in all_x])
    print(f"  {'|x_local| mean [cm]':<25} {m_dock_x:>12.2f} {m_all_x:>14.2f} {m_dock_x-m_all_x:>+10.2f}")

    # v_z buckets for success vs all
    print(f"\n  v_z distribution at first overlap:")
    vz_edges = [-2.0, -1.5, -1.0, -0.5, -0.3, 0.0]
    print(f"  {'Range':<20} {'Success':>10} {'All':>10}")
    prev = float('-inf')
    for edge in vz_edges:
        cnt_d = sum(1 for v in dock_vz if prev <= v < edge)
        cnt_a = sum(1 for v in all_vz if prev <= v < edge)
        pct_d = 100 * cnt_d / max(n_dock, 1)
        pct_a = 100 * cnt_a / max(n_all, 1)
        lo = f"{prev:.1f}" if prev != float('-inf') else "-inf"
        print(f"  [{lo:>5},{edge:>5})       {cnt_d:>4}({pct_d:>3.0f}%)  {cnt_a:>4}({pct_a:>3.0f}%)")
        prev = edge
    cnt_d = sum(1 for v in dock_vz if v >= prev)
    cnt_a = sum(1 for v in all_vz if v >= prev)
    print(f"  [{prev:>5}, +inf)       {cnt_d:>4}({100*cnt_d/max(n_dock,1):>3.0f}%)  {cnt_a:>4}({100*cnt_a/max(n_all,1):>3.0f}%)")

print(f"{'='*60}")

# Approach angle distribution
_percentile_buckets("Approach angle distribution",
                    diag["fail_approach_angle_deg"],
                    [15, 30, 45, 60, 75], "deg")

print(f"\n  --- Pre-failure 3-step history (means) ---")
def _mean_step(seqs, k):
    vals = [s[k] for s in seqs]
    return sum(vals) / len(vals) if vals else 0.0

if diag["pre_fail_vz_seq"]:
    n_fail = len(diag["pre_fail_vz_seq"])
    print(f"  Failure episodes analyzed: {n_fail}")
    print(f"  step:           t-2      t-1      t-0")
    print(f"  v_z   [m/s]:   {_mean_step(diag['pre_fail_vz_seq'],0):+.3f}  "
          f"{_mean_step(diag['pre_fail_vz_seq'],1):+.3f}  "
          f"{_mean_step(diag['pre_fail_vz_seq'],2):+.3f}")
    print(f"  tilt [deg]:    {_mean_step(diag['pre_fail_tilt_deg_seq'],0):6.1f}  "
          f"{_mean_step(diag['pre_fail_tilt_deg_seq'],1):6.1f}  "
          f"{_mean_step(diag['pre_fail_tilt_deg_seq'],2):6.1f}")
    print(f"  xy_err [m]:    {_mean_step(diag['pre_fail_xy_err_seq'],0):.3f}  "
          f"{_mean_step(diag['pre_fail_xy_err_seq'],1):.3f}  "
          f"{_mean_step(diag['pre_fail_xy_err_seq'],2):.3f}")

print(f"{'='*60}")

# === Save results to file ===
import json
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_name = os.path.basename(os.path.dirname(CKPT)) + "_" + os.path.basename(CKPT).replace(".pt", "")
box_type = "kinematic" if cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled else "dynamic"
save_name = f"{timestamp}_{ckpt_name}_{box_type}"

# Save to thesis_data structure
json_dir = os.path.join("thesis_data", "01_rl_experiments", "eval_raw")
csv_dir = os.path.join("thesis_data", "data_csv")
os.makedirs(json_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# JSON: all numerical data
json_data = {
    "timestamp": timestamp,
    "checkpoint": CKPT,
    "box_type": box_type,
    "episode_length_s": cfg.episode_length_s,
    "num_envs": cfg.scene.num_envs,
    "results": {k: v for k, v in results.items()},
    "diag": {k: v for k, v in diag.items()},
}
json_path = os.path.join(json_dir, f"{save_name}.json")
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2, default=str)

# TXT: redirect printed output
import io, sys as _sys
txt_path = os.path.join(json_dir, f"{save_name}.txt")
# Re-generate summary text
buf = io.StringIO()
_orig = _sys.stdout
_sys.stdout = buf

print(f"Checkpoint: {CKPT}")
print(f"Box type: {box_type}")
print(f"Episode length: {cfg.episode_length_s}s")
print(f"Num envs: {cfg.scene.num_envs}")
print(f"")
print(f"Total episodes:     {results['total_episodes']}")
print(f"Dock success:       {results['dock_success']}/{results['total_episodes']} "
      f"({100*results['dock_success']/max(results['total_episodes'],1):.1f}%)")
if results['dock_success'] > 0:
    avg_xy = sum(results['xy_err_at_dock']) / len(results['xy_err_at_dock'])
    print(f"Avg XY err at dock: {avg_xy*100:.1f}cm")
    print(f"Avg dock time:      {results['dock_time_sum']/results['dock_success']:.1f}s")
print(f"Grasp success:      {results['grasp_success']}/{results['total_episodes']} "
      f"({100*results['grasp_success']/max(results['total_episodes'],1):.1f}%)")
print(f"Crash:              {results.get('fail_crash', 0)}")
print(f"Far:                {results.get('fail_far', 0)}")
print(f"Below box:          {results.get('fail_below', 0)}")
print(f"Near timeout:       {results.get('fail_near_timeout', 0)}")
if diag.get("dock_bounce_count"):
    n_dock = len(diag["dock_bounce_count"])
    zero_b = sum(1 for b in diag["dock_bounce_count"] if b == 0)
    clean = sum(1 for b, t in zip(diag["dock_bounce_count"], diag["dock_tilt_deg"]) if b == 0 and t < 10)
    print(f"Zero-bounce docks:  {zero_b}/{n_dock} ({100*zero_b/max(n_dock,1):.0f}%)")
    print(f"Dynamic readiness:  {clean}/{n_dock} ({100*clean/max(n_dock,1):.0f}%)")
if diag.get("dock_first_ov_vz"):
    import statistics
    print(f"Success v_z@1st_ov: mean={sum(diag['dock_first_ov_vz'])/len(diag['dock_first_ov_vz']):.3f} "
          f"med={statistics.median(diag['dock_first_ov_vz']):.3f}")
if diag.get("first_overlap_vz"):
    import statistics
    print(f"All v_z@1st_ov:     mean={sum(diag['first_overlap_vz'])/len(diag['first_overlap_vz']):.3f} "
          f"med={statistics.median(diag['first_overlap_vz']):.3f}")

_sys.stdout = _orig
txt_content = buf.getvalue()

txt_path = os.path.join(json_dir, f"{save_name}.txt")
with open(txt_path, "w") as f:
    f.write(txt_content)

# CSV: append to rl_eval_results.csv for thesis comparison table
csv_path = os.path.join(csv_dir, "rl_eval_results.csv")
csv_exists = os.path.exists(csv_path)
import csv
with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    if not csv_exists:
        w.writerow(["timestamp", "checkpoint", "box_type", "episode_s", "total_episodes",
                     "dock_pct", "grasp_pct", "crash_pct", "far_pct", "near_timeout_pct",
                     "avg_xy_err_at_dock_cm", "zero_bounce_pct", "dynamic_readiness_pct",
                     "success_vz_median", "all_vz_median"])
    n_ep = max(results["total_episodes"], 1)
    dock_pct = 100 * results["dock_success"] / n_ep
    grasp_pct = 100 * results["grasp_success"] / n_ep
    crash_pct = 100 * results.get("fail_crash", 0) / n_ep
    far_pct = 100 * results.get("fail_far", 0) / n_ep
    near_to_pct = 100 * results.get("fail_near_timeout", 0) / n_ep
    avg_xy = (sum(results["xy_err_at_dock"]) / len(results["xy_err_at_dock"]) * 100
              if results["xy_err_at_dock"] else 0)
    zb_pct = (100 * sum(1 for b in diag.get("dock_bounce_count", []) if b == 0)
              / max(len(diag.get("dock_bounce_count", [])), 1)
              if diag.get("dock_bounce_count") else 0)
    dr_pct = (100 * sum(1 for b, t in zip(diag.get("dock_bounce_count", []),
              diag.get("dock_tilt_deg", [])) if b == 0 and t < 10)
              / max(len(diag.get("dock_bounce_count", [])), 1)
              if diag.get("dock_bounce_count") else 0)
    import statistics as _st
    s_vz = (_st.median(diag["dock_first_ov_vz"]) if diag.get("dock_first_ov_vz") else "")
    a_vz = (_st.median(diag["first_overlap_vz"]) if diag.get("first_overlap_vz") else "")
    w.writerow([timestamp, CKPT, box_type, cfg.episode_length_s, results["total_episodes"],
                f"{dock_pct:.1f}", f"{grasp_pct:.1f}", f"{crash_pct:.1f}",
                f"{far_pct:.1f}", f"{near_to_pct:.1f}",
                f"{avg_xy:.1f}", f"{zb_pct:.0f}", f"{dr_pct:.0f}",
                f"{s_vz:.3f}" if s_vz != "" else "", f"{a_vz:.3f}" if a_vz != "" else ""])

print(f"\n  Results saved to:")
print(f"    JSON: {json_path}")
print(f"    TXT:  {txt_path}")
print(f"    CSV:  {csv_path} (appended)")

env.close()
simulation_app.close()
