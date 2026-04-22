"""Quick test: does the box actually stay in the gripper during training?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = WaypointEnvCfg(mode="loaded")
cfg.scene.num_envs = 16

env = WaypointDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

obs, _ = env_wrapped.reset()

print("Checking box grip over 100 steps...")
for step in range(100):
    action = torch.zeros(16, 4, device=device)  # hover (zero action)
    obs, _, _, _, _ = env_wrapped.step(action)

    drone_pos = env.robot.data.root_pos_w
    box_pos = env.grasp_box.data.root_pos_w
    box_dist = torch.norm(drone_pos - box_pos, dim=-1)
    box_z = box_pos[:, 2]

    if step % 20 == 0:
        print(f"  step {step}: drone_z={drone_pos[0,2]:.2f} box_z={box_z[0]:.2f} "
              f"box_dist={box_dist[0]:.3f} "
              f"box_dropped={int((box_dist > 0.2).sum())} / {16}")

# Check final state
dropped = (box_dist > 0.2).sum().item()
print(f"\nResult: {dropped}/16 boxes dropped after 100 steps")
if dropped > 0:
    print("*** BOX IS FALLING FROM GRIPPER! Training is without payload! ***")
else:
    print("Box held securely.")

env.close()
simulation_app.close()
