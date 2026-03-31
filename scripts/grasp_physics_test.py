"""Test if grasping is physically real: does the box lift when drone ascends?"""
import sys, os
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
cfg.scene.num_envs = 64
cfg.episode_length_s = 40.0
env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
agent = build_ppo_agent(env=env_wrapped, device=env.device, stage=3,
    checkpoint_path="runs/26-03-25_04-41-48-354650_PPO/checkpoints/best_agent.pt")
agent.set_running_mode("eval")

obs, _ = env_wrapped.reset()

print("=== Grasp Physics Verification ===")
print("Testing: when 'grasped', does box z increase (lifted) or stay on ground (fake)?")
print()

total_steps = 6000
box_initial_z = env.object_pos[:, 2].clone()  # z=0.06 for all

# Track per-episode stats
episode_grasps = 0
real_lifts = 0
fake_grasps = 0

for step in range(total_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=total_steps)[0]
    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    # Check grasped envs
    grasped_mask = env.is_grasped
    if grasped_mask.any():
        grasped_ids = grasped_mask.nonzero(as_tuple=False).squeeze(-1)
        if grasped_ids.dim() == 0:
            grasped_ids = grasped_ids.unsqueeze(0)

        for idx in grasped_ids:
            i = idx.item()
            box_z = env.grasp_object.data.root_pos_w[i, 2].item()
            drone_z = env.robot.data.root_pos_w[i, 2].item()
            gripper_z = drone_z - 0.1485
            box_lifted = box_z > 0.10  # box center normally at 0.06, if > 0.10 it's lifted

            if step % 500 == 0 and i < 4:
                plate = env.robot.data.joint_pos[i, env.plate_joint_ids[0]].item()
                xy_off = torch.norm(env.robot.data.root_pos_w[i, :2] - env.grasp_object.data.root_pos_w[i, :2]).item()
                print(f"  step {step} env{i}: box_z={box_z:.3f} drone_z={drone_z:.3f} "
                      f"gripper_z={gripper_z:.3f} plate={plate:.3f}rad xy_off={xy_off:.3f}m "
                      f"{'LIFTED' if box_lifted else 'ON_GROUND'}")

    if terminated.any() or truncated.any():
        # Before reset, check all grasped envs
        reset_ids = (terminated | truncated).nonzero(as_tuple=False)
        if reset_ids.dim() > 1:
            reset_ids = reset_ids[:, 0]

        for idx in reset_ids:
            i = idx.item()
            if env.is_grasped[i].item():
                episode_grasps += 1
                box_z = env.grasp_object.data.root_pos_w[i, 2].item()
                if box_z > 0.10:
                    real_lifts += 1
                else:
                    fake_grasps += 1

        obs, _ = env_wrapped.reset()
        box_initial_z = env.object_pos[:, 2].clone()

# Final check: how many envs are grasped and box position
print()
print("=== Final State Analysis ===")
n_grasped = env.is_grasped.sum().item()
n_lifted = 0
n_ground = 0
for i in range(env.num_envs):
    if env.is_grasped[i].item():
        box_z = env.grasp_object.data.root_pos_w[i, 2].item()
        if box_z > 0.10:
            n_lifted += 1
        else:
            n_ground += 1

print(f"Currently grasped: {n_grasped}/{env.num_envs}")
print(f"  Box lifted (z > 0.10m): {n_lifted}")
print(f"  Box on ground (z <= 0.10m): {n_ground}")
print()
print(f"=== Episode-end Summary ===")
print(f"Episodes ending with grasp: {episode_grasps}")
print(f"  Real lifts (box lifted): {real_lifts}")
print(f"  Fake grasps (box on ground): {fake_grasps}")
if episode_grasps > 0:
    print(f"  Real grasp rate: {real_lifts/episode_grasps*100:.1f}%")

env.close()
simulation_app.close()
