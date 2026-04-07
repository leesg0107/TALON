"""Quick visual test: Stage 4 physical grasp + loaded flight."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.LOADED_FLIGHT)
cfg.scene.num_envs = 4
cfg.scene.env_spacing = 10.0
# Dynamic box
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

obs, _ = env_wrapped.reset()

print("\n=== Stage 4 Physical Grasp Test ===")
print("Phase 1: Drone hovers above box, gripper auto-closes")
print("Phase 2: Once grasped (chc>=100), check if box stays\n")

for step in range(4500):  # 30s
    # Neutral hover action: no lateral/vertical command, no rotation
    # action[0:3]=acceleration cmd, [3:6]=rate cmd, [6]=thrust, [7]=gripper
    action = torch.zeros(env.num_envs, 8, device=env.device)
    # Small negative z-accel to counteract any upward drift (hover)
    # action[:, 2] = -0.1  # uncomment if drone drifts up
    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    if step % 75 == 0:
        for i in range(env.num_envs):
            pos = env.robot.data.root_pos_w[i].cpu()
            obj = env.object_pos[i].cpu()
            plate = env.robot.data.joint_pos[i, env.plate_joint_ids[0]].item()
            chc = env.contain_hold_count[i].item()
            grasped = chc >= 100
            goal = env.goal_pos[i].cpu()
            # gripper-box distance
            gripper_z = pos[2].item() - 0.08
            box_z = obj[2].item()
            print(f"  env{i}: drone_z={pos[2]:.3f} box_z={box_z:.3f} "
                  f"grip_z={gripper_z:.3f} plate={plate:.3f}rad "
                  f"chc={chc} grasped={grasped} rew={reward[i].item():.2f}")
        print()

    if terminated.any() or truncated.any():
        print(f"  Reset at step {step}: term={terminated.sum().item()} trunc={truncated.sum().item()}")
        obs, _ = env_wrapped.reset()

env.close()
simulation_app.close()
