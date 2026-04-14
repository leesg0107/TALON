"""Rendered evaluation of PD analytical controller — visual inspection."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)  # RENDER ON
simulation_app = app_launcher.app

import torch
import time
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 4          # few envs for visual clarity
cfg.scene.env_spacing = 3.0
cfg.episode_length_s = 12.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0        # Pure analytical
# Dynamic box
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
# Agent needed for wrapper but output is ignored (analytical-only)
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/stage3_dynamic_v2/best_agent.pt")
agent.set_running_mode("eval")

from envs.drone_env import quat_to_rot_matrix

num_envs = env.num_envs
episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
total_steps = episode_steps * 5  # 5 episodes per env

print(f"\n=== Rendered PD Evaluation ===")
print(f"  Envs: {num_envs}, Episode: {episode_steps} steps ({cfg.episode_length_s}s)")
print(f"  Total: {total_steps} steps")
print(f"  Controls: watch the drone approach and dock\n")

obs, _ = env_wrapped.reset()
dock_count = 0
episode_count = 0

for step in range(total_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=total_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    # Compute metrics for display
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

    # Print status every 50 steps
    if step % 50 == 0:
        print(f"  step {step:>5}  xy={xy_err[0]:.3f}  z_off={z_offset[0]:.3f}  "
              f"vz={vz[0]:+.3f}  ov={overlap_ratio[0]:.2f}  "
              f"dock_hold={env.contain_hold_count[0].item()}/150")

    # Episode end
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat
    if done_mask.any():
        for i in done_mask.nonzero(as_tuple=False).view(-1).tolist():
            episode_count += 1
            docked = hasattr(env, '_dock_success') and env._dock_success[i].item()
            if docked:
                dock_count += 1
            status = "DOCK ✓" if docked else ("CRASH" if term_flat[i] and pos_w[i, 2].item() < 0.15 else "FAIL")
            print(f"  >>> Env {i} episode end: {status}  "
                  f"(dock_hold={env.contain_hold_count[i].item()}, "
                  f"xy={xy_err[i]:.3f}, ov={overlap_ratio[i]:.2f})")

        obs, _ = env_wrapped.reset()

print(f"\n{'='*50}")
print(f"  Episodes: {episode_count}, Dock: {dock_count} ({100*dock_count/max(episode_count,1):.1f}%)")
print(f"{'='*50}")

env.close()
simulation_app.close()
