"""Quick diagnostic: can WaypointDroneEnv run at all?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = WaypointEnvCfg(mode="flight")
cfg.scene.num_envs = 16  # small for debug
cfg.episode_length_s = 6.0

print("Creating env...")
env = WaypointDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
print(f"Env created. obs_space={env_wrapped.observation_space}, act_space={env_wrapped.action_space}")

print("Resetting...")
obs, _ = env_wrapped.reset()
print(f"Reset done. obs shape={obs.shape}, obs range=[{obs.min():.2f}, {obs.max():.2f}]")

print("Stepping 100 times with random actions...")
device = env.device
t0 = time.time()
for i in range(100):
    action = torch.randn(16, 4, device=device).clamp(-1, 1)
    obs, reward, term, trunc, info = env_wrapped.step(action)
    if i % 20 == 0:
        pos = env.robot.data.root_pos_w[0]
        print(f"  step {i}: pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
              f"reward={reward[0].item():.2f} term={term[0].item()} trunc={trunc[0].item()}")
dt = time.time() - t0
print(f"100 steps in {dt:.2f}s ({100/dt:.0f} steps/sec)")

print("\nDone. Env works.")
env.close()
simulation_app.close()
