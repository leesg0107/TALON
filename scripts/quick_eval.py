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
total_steps = 6000
grasped_steps = 0
total_grasped_envs = 0

for step in range(total_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=total_steps)[0]
    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    n_grasped = env.is_grasped.sum().item()
    grasped_steps += n_grasped
    total_grasped_envs += env.num_envs

    if step % 1000 == 0:
        pos = env.robot.data.root_pos_w
        obj = env.object_pos
        xy_err = torch.norm(pos[:, :2] - obj[:, :2], dim=-1).mean().item()
        alt_diff = (pos[:, 2] - obj[:, 2]).mean().item()
        plate = env.robot.data.joint_pos[:, env.plate_joint_ids[0]].mean().item()
        print(f"step {step}: grasped={n_grasped}/64  xy_err={xy_err:.3f}  alt_above_box={alt_diff:.3f}  plate={plate:.3f}rad  hold_so_far={grasped_steps/max(total_grasped_envs,1)*100:.1f}%")

    if terminated.any() or truncated.any():
        obs, _ = env_wrapped.reset()

print(f"\nTOTAL: hold={grasped_steps/total_grasped_envs*100:.1f}% over {total_steps} steps x 64 envs")
env.close()
simulation_app.close()
