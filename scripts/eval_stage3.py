"""Stage 3 evaluation: physics-based approach + grasp."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 4
cfg.scene.env_spacing = 10.0  # wider spacing for visualization
cfg.episode_length_s = 40.0
cfg.lock_gripper = False  # Allow gripper control for force-close test

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="runs/26-03-30_06-07-29-947064_PPO/checkpoints/best_agent.pt")
agent.set_running_mode("eval")

max_steps = 3000
grasp_count = 0
hold_steps = 0

print(f"\n=== Stage 3 Evaluation: Approach + Grasp ===\n")

obs, _ = env_wrapped.reset()

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    # Gripper control: always open, close only when sustained alignment
    if not hasattr(env, '_align_count'):
        env._align_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._gripper_closed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for i in range(env.num_envs):
        pos_i = env.robot.data.root_pos_w[i]
        obj_i = env.object_pos[i]
        xy_off = torch.norm(pos_i[:2] - obj_i[:2]).item()
        alt_diff = pos_i[2].item() - obj_i[2].item()
        in_range = xy_off < 0.15 and 0.05 < alt_diff < 0.20

        if env._gripper_closed[i]:
            if in_range:
                # Still close: keep gripper closed
                action[i, 7] = -1.0
            else:
                # Moved away: reopen gripper, reset
                action[i, 7] = 1.0
                env._gripper_closed[i] = False
                env._align_count[i] = 0
            continue

        # Default: force gripper fully open
        action[i, 7] = 1.0

        # Count sustained alignment
        if in_range:
            env._align_count[i] += 1
        else:
            env._align_count[i] = 0

        # Aligned for 30 steps (0.2s): close gripper
        if env._align_count[i] > 30:
            action[i, 7] = -1.0
            env._gripper_closed[i] = True

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    n_grasped = env.is_grasped.sum().item()
    hold_steps += n_grasped

    if step % 150 == 0:
        grasped_str = ""
        for i in range(min(env.num_envs, 4)):
            pos_i = env.robot.data.root_pos_w[i].cpu()
            obj_i = env.object_pos[i].cpu()
            d = torch.norm(pos_i - obj_i).item()
            xy_off = torch.norm(pos_i[:2] - obj_i[:2]).item()
            z_above = (pos_i[2] - obj_i[2]).item()
            plate = env.robot.data.joint_pos[i, env.plate_joint_ids[0]].item()
            g = "G" if env.is_grasped[i].item() else "."
            box_z = obj_i[2].item()
            grasped_str += f"  env{i}:d={d:.2f} xy={xy_off:.2f} z+={z_above:.2f} plt={plate:.2f} bz={box_z:.3f}{g}"
        print(f"  step {step:4d} | grasped={n_grasped}/{env.num_envs}{grasped_str}")

    if terminated.any() or truncated.any():
        grasp_count += n_grasped
        print(f"  --- Episode reset at step {step} (grasped={n_grasped}/{env.num_envs}) ---")
        obs, _ = env_wrapped.reset()

print(f"\n=== Results ===")
print(f"  Grasp successes: {grasp_count}")
total_env_steps = max_steps * cfg.scene.num_envs
print(f"  Hold: {hold_steps}/{total_env_steps} ({100*hold_steps/total_env_steps:.1f}%)")

env.close()
simulation_app.close()
