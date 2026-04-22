"""Debug: compare obs/action between GripperWaypointEnv flow vs manual obs flow.
Same env, same agent, two different obs computation methods."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.gripper_waypoint_env import GripperWaypointEnv
from envs.drone_env import quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from envs.waypoint_cfg import WaypointEnvCfg

CKPT = "logs/gripper_wp_flight_v2/best_agent.pt"

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 20.0
cfg.lock_gripper = True
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperWaypointEnv(cfg=cfg, mode="flight")
env_w = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_w, device=device, mode="flight",
                             checkpoint_path=CKPT)
agent.set_running_mode("eval")

# Manual obs (same as eval_mission compute_waypoint_obs)
def compute_obs_manual(env, goal_pos, prev_action_4d):
    vel_b = env.robot.data.root_lin_vel_b[0]
    ang_vel_b = env.robot.data.root_ang_vel_b[0]
    quat_w = env.robot.data.root_quat_w
    R = quat_to_rot_matrix(quat_w)
    pos_w = env.robot.data.root_pos_w[0]
    goal_err_w = goal_pos - pos_w
    goal_b = R[0].T @ goal_err_w
    R_flat = R[0].reshape(9)
    prev_norm = prev_action_4d.clone()
    prev_norm[:3] /= 8.0
    prev_norm[3] /= math.pi
    obs = torch.cat([vel_b, ang_vel_b, R_flat, goal_b, prev_norm])
    return obs.unsqueeze(0)

obs_env, _ = env_w.reset()
goal = env.goal_pos[0].clone()
prev_4d = torch.zeros(4, device=device)

print(f"\n{'='*80}")
print(f"  Comparing env obs vs manual obs (same env, same state)")
print(f"  Goal: {goal.tolist()}")
print(f"  Pos:  {env.robot.data.root_pos_w[0].tolist()}")
print(f"{'='*80}\n")

for step in range(5):
    # Method A: obs from env (what standalone uses)
    # obs_env is from previous step's env_w.step() or reset()

    # Method B: manual obs (what eval_mission uses)
    obs_manual = compute_obs_manual(env, goal, prev_4d)

    # Compare
    diff = (obs_env - obs_manual).abs()
    max_diff = diff.max().item()

    print(f"Step {step}:")
    print(f"  Env obs[0:3] (vel_b):   {obs_env[0, 0:3].tolist()}")
    print(f"  Manual obs[0:3] (vel_b): {obs_manual[0, 0:3].tolist()}")
    print(f"  Env obs[15:18] (goal_b): {obs_env[0, 15:18].tolist()}")
    print(f"  Manual obs[15:18] (goal_b): {obs_manual[0, 15:18].tolist()}")
    print(f"  Env obs[18:22] (prev_act): {obs_env[0, 18:22].tolist()}")
    print(f"  Manual obs[18:22] (prev_act): {obs_manual[0, 18:22].tolist()}")
    print(f"  MAX DIFF: {max_diff:.6f}")

    # Get action from BOTH obs
    with torch.no_grad():
        action_from_env = agent.act(obs_env, timestep=step, timesteps=100)[0]
        action_from_manual = agent.act(obs_manual, timestep=step, timesteps=100)[0]

    act_diff = (action_from_env - action_from_manual).abs().max().item()
    print(f"  Action from env obs:    {action_from_env[0].tolist()}")
    print(f"  Action from manual obs: {action_from_manual[0].tolist()}")
    print(f"  ACTION DIFF: {act_diff:.6f}")
    print()

    # Step with env-obs action
    prev_4d = action_from_env[0].clone()
    obs_env, _, _, _, _ = env_w.step(action_from_env)

    # Update goal (might have changed due to regen)
    goal = env.goal_pos[0].clone()

env.close()
simulation_app.close()
