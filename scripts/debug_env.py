"""Debug: check xy_err in 4096 envs with pedestal."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 4096

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

obs, _ = env_wrapped.reset()

# Check positions
pos_w = env.robot.data.root_pos_w
obj_pos = env.object_pos
gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=env.device)
R = quat_to_rot_matrix(env.robot.data.root_quat_w)
gripper_pos_w = pos_w + torch.bmm(R, gripper_offset.expand(4096, -1).unsqueeze(-1)).squeeze(-1)

xy_err = torch.norm(gripper_pos_w[:, :2] - obj_pos[:, :2], dim=-1)

print(f"Drone pos_w: mean={pos_w.mean(0).cpu().numpy()}, std={pos_w.std(0).cpu().numpy()}")
print(f"Object pos:  mean={obj_pos.mean(0).cpu().numpy()}, std={obj_pos.std(0).cpu().numpy()}")
print(f"Gripper pos: mean={gripper_pos_w.mean(0).cpu().numpy()}")
print(f"XY err:      mean={xy_err.mean().item():.4f}, max={xy_err.max().item():.4f}, min={xy_err.min().item():.4f}")
print()

# Check a few specific envs
for i in [0, 100, 1000, 4000]:
    d = pos_w[i].cpu().numpy()
    o = obj_pos[i].cpu().numpy()
    g = gripper_pos_w[i].cpu().numpy()
    print(f"env{i}: drone={d}, box={o}, gripper={g}, xy_err={xy_err[i].item():.4f}")

env.close()
simulation_app.close()
