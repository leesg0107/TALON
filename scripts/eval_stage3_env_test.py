"""Stage 3 environment test: run Stage 2 model in Stage 3 env WITHOUT training.
Diagnoses whether approach degrades from environment setup alone vs training."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Stage 3 environment with gripper unlock t=0 (locked, same as Stage 2)
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 64
cfg.episode_length_s = 12.0

env = GripperDroneEnv(cfg=cfg)
env.gripper_unlock_t = 0.0  # locked
env_wrapped = SkrlVecEnvWrapper(env)

device = env.device
agent = build_ppo_agent(env=env_wrapped, device=device, stage=3,
                        checkpoint_path="runs/26-03-23_23-06-40-055753_PPO/checkpoints/best_agent.pt")
agent.set_running_mode("eval")

max_steps = 1800  # 12s at 150Hz
print(f"\n=== Stage 3 Env Test (Stage 2 model, NO training) ===")
print(f"  gripper_unlock_t = 0.0 (locked)")
print(f"  Spawn: z=3.0 (same as Stage 2)")
print(f"  Envs: {cfg.scene.num_envs}\n")

obs, _ = env_wrapped.reset()

xy_errs = []
alt_errs = []
d_objs = []

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    pos = env.robot.data.root_pos_w  # (N, 3)
    goal = env.goal_pos  # (N, 3)
    obj = env.object_pos  # (N, 3)

    xy_err = torch.norm(pos[:, :2] - goal[:, :2], dim=-1).mean().item()
    alt_err = torch.abs(pos[:, 2] - goal[:, 2]).mean().item()
    d_obj = torch.norm(pos - obj, dim=-1).mean().item()

    xy_errs.append(xy_err)
    alt_errs.append(alt_err)
    d_objs.append(d_obj)

    if step % 300 == 0:
        print(f"  step {step:4d} | xy_err={xy_err:.3f}m | alt_err={alt_err:.3f}m | d_obj={d_obj:.3f}m")

    if terminated.any() or truncated.any():
        obs, _ = env_wrapped.reset()

# Summary
n = len(xy_errs)
print(f"\n=== Summary (over {max_steps} steps, {cfg.scene.num_envs} envs) ===")
print(f"  xy_err:  first={xy_errs[0]:.3f}  mid={xy_errs[n//2]:.3f}  last={xy_errs[-1]:.3f}")
print(f"  alt_err: first={alt_errs[0]:.3f}  mid={alt_errs[n//2]:.3f}  last={alt_errs[-1]:.3f}")
print(f"  d_obj:   first={d_objs[0]:.3f}  mid={d_objs[n//2]:.3f}  last={d_objs[-1]:.3f}")

# Also test with gripper_unlock_t = 1.0
print(f"\n=== Now testing with gripper_unlock_t = 1.0 (fully unlocked) ===\n")
env.gripper_unlock_t = 1.0
obs, _ = env_wrapped.reset()

xy_errs2 = []
d_objs2 = []

for step in range(max_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)

    pos = env.robot.data.root_pos_w
    goal = env.goal_pos
    obj = env.object_pos

    xy_err = torch.norm(pos[:, :2] - goal[:, :2], dim=-1).mean().item()
    d_obj = torch.norm(pos - obj, dim=-1).mean().item()
    xy_errs2.append(xy_err)
    d_objs2.append(d_obj)

    if step % 300 == 0:
        print(f"  step {step:4d} | xy_err={xy_err:.3f}m | d_obj={d_obj:.3f}m")

    if terminated.any() or truncated.any():
        obs, _ = env_wrapped.reset()

n2 = len(xy_errs2)
print(f"\n=== Summary (unlocked gripper) ===")
print(f"  xy_err:  first={xy_errs2[0]:.3f}  mid={xy_errs2[n2//2]:.3f}  last={xy_errs2[-1]:.3f}")
print(f"  d_obj:   first={d_objs2[0]:.3f}  mid={d_objs2[n2//2]:.3f}  last={d_objs2[-1]:.3f}")

env.close()
simulation_app.close()
