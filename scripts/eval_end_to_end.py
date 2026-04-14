"""End-to-end mission: waypoint approach → dock → loaded delivery.

Phase 1: Stage 1 RL flies to waypoint above box (safe altitude)
Phase 2: PD analytical descends and docks with box
Phase 3: Stage 4 RL flies to delivery waypoint with grasped box

Single env, rendered, 10 episodes with auto-reset.
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from agents.ppo_cfg import build_ppo_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ---- Config ----
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 120.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False
cfg.spawn_spread = 0.0

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

# ---- Load models ----
STAGE1_CKPT = "logs/stage1_ppo_v2/best_agent.pt"
STAGE4_CKPT = "logs/stage4_pre_grasped_v9/final_agent.pt"

agent_s1 = build_ppo_agent(env=env_wrapped, device=device, stage=1,
                           checkpoint_path=STAGE1_CKPT)
agent_s1.set_running_mode("eval")

agent_s4 = build_ppo_agent(env=env_wrapped, device=device, stage=4,
                           checkpoint_path=STAGE4_CKPT)
agent_s4.set_running_mode("eval")

# ---- Mission config ----
NUM_MISSIONS = 10
MAX_STEPS_PER_MISSION = 18000  # 120s
WP_ARRIVE_DIST = 0.40
DOCK_XY_SWITCH = 0.50
DOCK_THRESHOLD = 225
GRIP_SETTLE_STEPS = 150
DELIVERY_ARRIVE = 0.40

# Random start positions for variety
START_POSITIONS = [
    [2.0, 2.0, 3.0], [-2.0, 1.5, 3.5], [1.5, -2.0, 3.0],
    [-1.0, 2.5, 3.0], [2.5, -1.0, 3.5], [-2.0, -2.0, 3.0],
    [1.0, 2.0, 3.5], [-1.5, -1.5, 3.0], [2.0, 0.5, 3.0],
    [-0.5, 2.0, 3.5],
]

DELIVERY_GOALS = [
    [1.5, -1.5, 2.0], [-1.5, 1.5, 2.0], [2.0, 1.0, 2.5],
    [-1.0, -2.0, 2.0], [1.0, 2.0, 2.0], [-2.0, 0.5, 2.5],
    [0.5, -2.0, 2.0], [1.5, 1.5, 2.0], [-1.5, -0.5, 2.5],
    [2.0, -1.0, 2.0],
]

# Results
mission_results = []

print(f"\n{'='*60}")
print(f"  END-TO-END MISSION DEMO ({NUM_MISSIONS} episodes)")
print(f"  Stage 1: {STAGE1_CKPT}")
print(f"  Stage 4: {STAGE4_CKPT}")
print(f"{'='*60}\n")

obs, _ = env_wrapped.reset()

for mission in range(NUM_MISSIONS):
    # ---- Setup mission ----
    start_pos = torch.tensor(START_POSITIONS[mission], device=device)
    wp_delivery = torch.tensor(DELIVERY_GOALS[mission], device=device)

    # Read box position
    box_pos = env.object_pos[0].clone()

    # Teleport drone to start
    root_state = torch.zeros(1, 13, device=device)
    root_state[0, :3] = start_pos
    root_state[0, 3] = 1.0  # identity quaternion
    env.robot.write_root_state_to_sim(root_state)

    # Reset box to pedestal
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = torch.tensor([0.0, 0.0, 0.54], device=device)
    obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([0], device=device))

    # Reset env state
    env.contain_hold_count[0] = 0
    if hasattr(env, '_stuck_count'):
        env._stuck_count[0] = 0
    env.is_grasped[0] = False
    env.step_count[0] = 0

    # Waypoint above box
    wp_above = torch.tensor([box_pos[0].item(), box_pos[1].item(), 2.0], device=device)

    # State
    phase = "approach"
    dock_count = 0
    dock_step = 0
    phase_step = 0
    mission_start_step = 0

    print(f"\n  --- Mission {mission+1}/{NUM_MISSIONS} ---")
    print(f"  Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
    print(f"  Delivery: ({wp_delivery[0]:.1f}, {wp_delivery[1]:.1f}, {wp_delivery[2]:.1f})")

    env.goal_pos[0] = wp_above
    env.bypass_analytical = True

    mission_success = False

    for step in range(MAX_STEPS_PER_MISSION):
        # Metrics
        pos_w = env.robot.data.root_pos_w[0]
        vel_w = env.robot.data.root_lin_vel_w[0]
        R = quat_to_rot_matrix(env.robot.data.root_quat_w)
        gripper_pos_w = pos_w + (R[0] @ torch.tensor([0., 0., -0.08], device=device))
        obj_pos = env.object_pos[0]
        xy_to_box = torch.norm(gripper_pos_w[:2] - obj_pos[:2]).item()

        # Overlap (simplified)
        box_local = R[0].T @ (obj_pos - gripper_pos_w)
        bh = 0.04
        ox = max(0, min(0.05, box_local[0].item()+bh) - max(-0.05, box_local[0].item()-bh))
        oy = max(0, min(0.062, box_local[1].item()+bh) - max(-0.062, box_local[1].item()-bh))
        z_ok = -0.12 < box_local[2].item() < 0.02
        overlap = ox * oy / (2*bh)**2 if z_ok else 0.0

        # ---- Phase logic ----
        if phase == "approach":
            env.bypass_analytical = True
            env.goal_pos[0] = wp_above
            with torch.no_grad():
                action = agent_s1.act(obs, timestep=step, timesteps=MAX_STEPS_PER_MISSION)[0]

            dist_wp = torch.norm(gripper_pos_w - wp_above).item()
            if dist_wp < WP_ARRIVE_DIST or xy_to_box < DOCK_XY_SWITCH:
                phase = "dock"
                phase_step = step
                env.bypass_analytical = False
                print(f"    [{step/150:.1f}s] APPROACH → DOCK (xy_box={xy_to_box:.2f})")

        elif phase == "dock":
            env.bypass_analytical = False
            with torch.no_grad():
                action = agent_s1.act(obs, timestep=step, timesteps=MAX_STEPS_PER_MISSION)[0]

            if overlap > 0.5:
                dock_count += 1
            if env.contain_hold_count[0] > 200:
                env.contain_hold_count[0] = 200

            if dock_count >= DOCK_THRESHOLD:
                phase = "grip_wait"
                dock_step = step
                print(f"    [{step/150:.1f}s] DOCK → GRIP_WAIT (time={(step-phase_step)/150:.1f}s)")

        elif phase == "grip_wait":
            with torch.no_grad():
                action = agent_s1.act(obs, timestep=step, timesteps=MAX_STEPS_PER_MISSION)[0]
            if env.contain_hold_count[0] > 200:
                env.contain_hold_count[0] = 200

            if step - dock_step >= GRIP_SETTLE_STEPS:
                phase = "flight"
                phase_step = step
                env.bypass_analytical = True
                drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[0]
                env.attitude_ctrl.mass[0] = drone_mass + 0.2
                env.goal_pos[0] = wp_delivery
                print(f"    [{step/150:.1f}s] GRIP_WAIT → FLIGHT")

        elif phase == "flight":
            env.bypass_analytical = True
            env.goal_pos[0] = wp_delivery
            with torch.no_grad():
                action = agent_s4.act(obs, timestep=step, timesteps=MAX_STEPS_PER_MISSION)[0]

            if env.contain_hold_count[0] > 200:
                env.contain_hold_count[0] = 200

            # Teleport box to gripper
            quat_w = env.robot.data.root_quat_w
            bs = torch.zeros(1, 13, device=device)
            bs[0, :3] = gripper_pos_w
            bs[0, 3:7] = quat_w[0]
            env.grasp_object.write_root_state_to_sim(bs, torch.tensor([0], device=device))

            dist_del = torch.norm(gripper_pos_w - wp_delivery).item()
            if dist_del < DELIVERY_ARRIVE:
                phase = "arrived"
                flight_time = (step - phase_step) / 150.0
                mission_success = True
                print(f"    [{step/150:.1f}s] *** ARRIVED *** (flight={flight_time:.1f}s)")

        elif phase == "arrived":
            with torch.no_grad():
                action = agent_s4.act(obs, timestep=step, timesteps=MAX_STEPS_PER_MISSION)[0]
            if env.contain_hold_count[0] > 200:
                env.contain_hold_count[0] = 200
            bs = torch.zeros(1, 13, device=device)
            bs[0, :3] = gripper_pos_w
            bs[0, 3:7] = env.robot.data.root_quat_w[0]
            env.grasp_object.write_root_state_to_sim(bs, torch.tensor([0], device=device))
            # Stay a moment then end
            if step - phase_step > 150:  # 1s hover at goal
                break

        obs, reward, terminated, truncated, info = env_wrapped.step(action)

        # Status
        if step % 150 == 0:
            extra = f"dock={dock_count}/{DOCK_THRESHOLD}" if phase == "dock" else ""
            print(f"    [{step/150:5.1f}s] {phase:>10}  pos=({pos_w[0]:.1f},{pos_w[1]:.1f},{pos_w[2]:.1f})  xy_box={xy_to_box:.2f}  {extra}")

        # Unexpected termination
        if (terminated.view(-1) | truncated.view(-1)).any():
            if phase != "arrived":
                tilt_deg = torch.acos(R[0, 2, 2].clamp(-1, 1)).item() * 57.3
                print(f"    [{step/150:.1f}s] TERMINATED at phase={phase} tilt={tilt_deg:.0f}° alt={pos_w[2]:.2f}")
            break

    total_time = step / 150.0
    result = "SUCCESS" if mission_success else f"FAIL ({phase})"
    mission_results.append(mission_success)
    print(f"  Result: {result} ({total_time:.1f}s)")

# ---- Summary ----
n_success = sum(mission_results)
print(f"\n{'='*60}")
print(f"  MISSION SUMMARY: {n_success}/{NUM_MISSIONS} complete ({100*n_success/NUM_MISSIONS:.0f}%)")
for i, r in enumerate(mission_results):
    print(f"    Mission {i+1}: {'SUCCESS' if r else 'FAIL'}")
print(f"{'='*60}")

env.close()
simulation_app.close()
