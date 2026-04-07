"""Debug Stage 4: analyze what causes rotation/instability."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import math
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = GripperDroneEnvCfg(stage=Stage.LOADED_FLIGHT)
cfg.scene.num_envs = 16
cfg.scene.env_spacing = 10.0
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)

# Load best model
ckpt = "runs/26-04-07_03-35-18-162561_PPO/checkpoints/best_agent.pt"
agent = build_ppo_agent(env=env_wrapped, device=env.device, stage=4, checkpoint_path=ckpt)
agent.set_running_mode("eval")

obs, _ = env_wrapped.reset()

print("\n=== Stage 4 Rotation Debug ===")
print("Columns: step | drone_z | tilt_deg | yaw_deg | speed | "
      "action[ax,ay,az,wx,wy,wz,yaw_ref] | box_dist\n")

from controllers.drone_ctrl import quat_to_rot_matrix

for step in range(1500):  # 10s
    with torch.no_grad():
        actions = agent.act(obs, timestep=0, timesteps=0)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(actions)

    if step % 30 == 0:  # every 0.2s
        pos = env.robot.data.root_pos_w
        quat = env.robot.data.root_quat_w
        vel = env.robot.data.root_lin_vel_w
        ang_vel = env.robot.data.root_ang_vel_b
        R = quat_to_rot_matrix(quat)

        # Tilt angle
        tilt_rad = torch.acos(R[:, 2, 2].clamp(-1, 1))
        tilt_deg = tilt_rad * 180 / math.pi

        # Yaw from rotation matrix
        yaw_rad = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        yaw_deg = yaw_rad * 180 / math.pi

        # Speed
        speed = torch.norm(vel, dim=-1)

        # Scaled actions
        scaled = env.scaled_actions

        # Box distance from gripper
        gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=env.device)
        gripper_pos = pos + torch.bmm(R, gripper_offset.expand(env.num_envs, 3).unsqueeze(-1)).squeeze(-1)
        box_dist = torch.norm(env.object_pos - gripper_pos, dim=-1)

        grasped = env.contain_hold_count >= 100

        for i in range(min(4, env.num_envs)):
            if not grasped[i]:
                phase = "GRASP"
            else:
                phase = "FLY  "
            a = scaled[i]
            print(f"  [{phase}] env{i} step={step:4d} z={pos[i,2]:.2f} "
                  f"tilt={tilt_deg[i]:.1f}° yaw={yaw_deg[i]:.1f}° "
                  f"spd={speed[i]:.2f} "
                  f"a=[{a[0]:.1f},{a[1]:.1f},{a[2]:.1f},{a[3]:.1f},{a[4]:.1f},{a[5]:.1f},{a[6]:.2f}] "
                  f"box_d={box_dist[i]:.3f} "
                  f"ang_vel=[{ang_vel[i,0]:.1f},{ang_vel[i,1]:.1f},{ang_vel[i,2]:.1f}]")
        print()

    if terminated.any() or truncated.any():
        done_ids = (terminated | truncated).squeeze().nonzero(as_tuple=False).squeeze(-1)
        for i in done_ids[:4]:
            t = "TERM" if terminated.squeeze()[i] else "TRUNC"
            print(f"  >>> env{i.item()} {t} at step {step}, "
                  f"z={pos[i,2]:.2f}, tilt={tilt_deg[i]:.1f}°, "
                  f"box_d={box_dist[i]:.3f}")
        obs, _ = env_wrapped.reset()

env.close()
simulation_app.close()
