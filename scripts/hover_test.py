"""Quick sanity test: does the drone hover with zero RL action?

If the inner-loop controller works, zero action (accel=0, rate=0, yaw=0)
should produce hover at the spawn position (z≈3.0m).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv

cfg = GripperDroneEnvCfg(stage=Stage.BASIC_FLIGHT)
cfg.scene.num_envs = 4
env = GripperDroneEnv(cfg=cfg)

# Zero action = hover command
zero_action = torch.zeros(4, 7, device=env.device)

print("\n=== HOVER TEST: stepping 200 times with zero action ===\n")

for i in range(200):
    obs, reward, terminated, truncated, info = env.step(zero_action)
    pos = env.robot.data.root_pos_w[0].cpu()
    vel = env.robot.data.root_lin_vel_w[0].cpu()

    if i % 25 == 0 or i < 5:
        print(f"  step {i:3d}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
              f"vel=[{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]  "
              f"reward={reward[0].item():.3f}  "
              f"terminated={terminated[0].item()}")

    if terminated[0]:
        print(f"\n  DRONE CRASHED at step {i}!")
        break

z_final = env.robot.data.root_pos_w[0, 2].item()
print(f"\nFinal position z={z_final:.3f}m (started at ~3.0m)")
if z_final > 2.0:
    print("RESULT: PASS - Drone maintains altitude")
else:
    print("RESULT: FAIL - Drone lost altitude or crashed")

env.close()
simulation_app.close()
