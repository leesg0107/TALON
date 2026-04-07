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
cfg.episode_length_s = 40.0
cfg.lock_gripper = False  # Allow gripper control

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/best_models/best_fc10_saturate.pt")
# NOTE: This is the best model with dock 56.5%, full_contain 10.8%
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

            # Docked after cumulative ~2s (300 steps)
            if align_count[i] > 300:
                action[i, 7] = -1.0
                gripper_closed[i] = True
                if not docked[i]:
                    docked[i] = True
                    dock_step[i] = env_step_count[i].clone()
                    results["dock_success"] += 1
                    dt = env_step_count[i].item() / 150.0
                    results["dock_time_sum"] += dt
                    results["xy_err_at_dock"].append(xy_err[i].item())

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
            if not is_docked:
                drone_z = pos_w[i, 2].item()
                box_z = env.object_pos[i, 2].item()
                xy = xy_err[i].item()
                is_term = term_flat[i].item()

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

            align_count[i] = 0
            gripper_closed[i] = False
            docked[i] = False
            dock_step[i] = -1
            grasped[i] = False
            grasp_step[i] = -1
            env_step_count[i] = 0
        obs, _ = env_wrapped.reset()

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

env.close()
simulation_app.close()
