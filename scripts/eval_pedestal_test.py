"""Quick visual test: pedestal + box environment."""
import sys, os
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
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 30.0

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="logs/best_models/best_fc11_fine.pt")
agent.set_running_mode("eval")

obs, _ = env_wrapped.reset()

for step in range(4500):  # 30s
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=4500)[0]
    action[:, 7] = 1.0  # gripper open
    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    if step % 150 == 0:
        for i in range(min(env.num_envs, 4)):
            pos = env.robot.data.root_pos_w[i].cpu()
            obj = env.object_pos[i].cpu()
            print(f"  env{i}: drone=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] box=[{obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f}]")
        print()

    if terminated.any() or truncated.any():
        print(f"  Reset at step {step}")
        obs, _ = env_wrapped.reset()

env.close()
simulation_app.close()
