"""Pre-simulate PD dock+climb to generate realistic grasp states for Stage 4 training.

Runs the PD controller (same as end-to-end eval) to dock the drone onto the box,
close the gripper, and climb. Records the final physical state (positions, velocities,
joint angles) for each successful grasp.

Output: data/grasp_states.pt — a dict of tensors, each [N_success, ...].
Stage 4 training loads these and samples randomly in _setup_grasped_state,
giving the RL agent physically-realistic grasped states without dock steps in the buffer.
"""
import sys, os, math
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ============================================================
# Config — same as end-to-end dock phase
# ============================================================
NUM_ENVS = 128
TARGET_STATES = 5000  # number of successful grasp states to collect
DOCK_TIMEOUT = 1800   # 12s at 150Hz
CLIMB_TARGET_Z = 1.2  # collect state when drone reaches this height

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = NUM_ENVS
cfg.scene.env_spacing = 10.0
cfg.episode_length_s = 30.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.goal_pos_range_xy = 0.0
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

# Disable auto-reset
def _no_dones():
    return (torch.zeros(NUM_ENVS, dtype=torch.bool, device=device),
            torch.zeros(NUM_ENVS, dtype=torch.bool, device=device))
env._get_dones = _no_dones

# ============================================================
# Per-env state
# ============================================================
# Phases: 0=dock, 1=climb, 2=capture_state, 3=done
phase = torch.full((NUM_ENVS,), 3, dtype=torch.long, device=device)
dock_step = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)

# Storage for captured states
captured = {
    'drone_pos_local': [],    # (3,) relative to env_origin
    'drone_quat': [],         # (4,) wxyz
    'drone_vel': [],          # (6,) lin + ang
    'box_pos_local': [],      # (3,) relative to env_origin
    'box_quat': [],           # (4,)
    'joint_pos': [],          # (num_joints,)
    'mass_scale': [],         # (1,)
    'payload_mass': [],       # (1,)
    'motor_kf_scale': [],     # (1,)
}

def setup_dock(eids):
    """Setup dock attempt — drone spawns above box with realistic entry conditions."""
    for eid in eids.tolist():
        origin = env.scene.env_origins[eid]
        box_pos = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)

        # Place box on pedestal
        obj_state = torch.zeros(1, 13, device=device)
        obj_state[0, :3] = box_pos
        obj_state[0, 3] = 1.0
        env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
        env.object_pos[eid] = box_pos

        # Randomized entry: ±8cm XY, 0.3-0.5m above, ±0.2 m/s
        xy_off = 0.16 * (torch.rand(2, device=device) - 0.5)   # ±8cm
        z_above = 0.30 + 0.20 * torch.rand(1, device=device).item()
        vel_xy = 0.40 * (torch.rand(2, device=device) - 0.5)   # ±0.2 m/s

        root_state = torch.zeros(1, 13, device=device)
        root_state[0, 0] = box_pos[0] + xy_off[0]
        root_state[0, 1] = box_pos[1] + xy_off[1]
        root_state[0, 2] = box_pos[2] + z_above
        root_state[0, 3] = 1.0
        root_state[0, 7:9] = vel_xy
        env.robot.write_root_state_to_sim(root_state, torch.tensor([eid], device=device))

        # Reset state
        env.contain_hold_count[eid] = 0
        env.step_count[eid] = 0
        if hasattr(env, '_stuck_count'):
            env._stuck_count[eid] = 0
        if hasattr(env, '_recovery_timer'):
            env._recovery_timer[eid] = 0
        if hasattr(env, '_z_integral'):
            env._z_integral[eid] = 0.0
        if hasattr(env, '_dock_hold_z'):
            env._dock_hold_z[eid] = 0.0
        env.attitude_ctrl.reset(torch.tensor([eid], device=device))

        phase[eid] = 0  # dock
        dock_step[eid] = 0

    env.reset_buf[eids] = False
    env.reset_terminated[eids] = False
    env.reset_time_outs[eids] = False

def capture_state(eid):
    """Capture the physical state of a successfully grasped+climbed drone."""
    origin = env.scene.env_origins[eid]
    pos_w = env.robot.data.root_pos_w[eid]
    quat_w = env.robot.data.root_quat_w[eid]
    vel = torch.cat([env.robot.data.root_lin_vel_w[eid],
                     env.robot.data.root_ang_vel_w[eid]])
    box_pos_w = env.grasp_object.data.root_pos_w[eid]
    box_quat = env.grasp_object.data.root_quat_w[eid]
    joint_pos = env.robot.data.joint_pos[eid]

    captured['drone_pos_local'].append((pos_w - origin).cpu())
    captured['drone_quat'].append(quat_w.cpu())
    captured['drone_vel'].append(vel.cpu())
    captured['box_pos_local'].append((box_pos_w - origin).cpu())
    captured['box_quat'].append(box_quat.cpu())
    captured['joint_pos'].append(joint_pos.cpu())
    captured['mass_scale'].append(env.mass_scale[eid:eid+1].cpu())
    captured['payload_mass'].append(torch.tensor([0.2]))
    captured['motor_kf_scale'].append(env.motor_kf_scale[eid:eid+1].cpu())

# ============================================================
# Init
# ============================================================
obs, _ = env_wrapped.reset()
all_ids = torch.arange(NUM_ENVS, device=device)
setup_dock(all_ids)

# Settle
env.bypass_analytical = False
for _ in range(3):
    env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))
    env.contain_hold_count[:] = 0

# Re-place boxes
for eid in range(NUM_ENVS):
    origin = env.scene.env_origins[eid]
    bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
    obj_state = torch.zeros(1, 13, device=device)
    obj_state[0, :3] = bp
    obj_state[0, 3] = 1.0
    env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
    env.object_pos[eid] = bp
env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))
env.contain_hold_count[:] = 0
phase[:] = 0

n_captured = 0
n_attempts = 0
n_dock_timeout = 0
n_box_fell = 0
n_climb_fail = 0
n_false_grasp = 0
print(f"\n{'='*60}")
print(f"  GRASP STATE GENERATION ({NUM_ENVS} envs, target={TARGET_STATES})")
print(f"{'='*60}\n")

# ============================================================
# Main loop
# ============================================================
for step in range(500000):
    if n_captured >= TARGET_STATES:
        break

    pos_w = env.robot.data.root_pos_w
    obj_pos = env.object_pos

    env.bypass_analytical = False
    env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))

    for eid in range(NUM_ENVS):
        if phase[eid] == 3:  # done, waiting for recycle
            continue

        contain = env.contain_hold_count[eid].item()
        cur_z = pos_w[eid, 2].item()
        origin_z = env.scene.env_origins[eid, 2].item()
        box_z_l = obj_pos[eid, 2].item() - origin_z
        bd = torch.norm(pos_w[eid] - obj_pos[eid]).item()

        if phase[eid] == 0:  # dock
            dock_step[eid] += 1

            # Dock success: transition to climb
            if contain >= 325:
                if bd < 0.30:
                    # Update mass for loaded flight
                    drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[eid]
                    env.attitude_ctrl.mass[eid] = drone_mass + 0.2
                    phase[eid] = 1  # climb
                else:
                    # False grasp, retry
                    n_false_grasp += 1
                    env.contain_hold_count[eid] = 0
                    if hasattr(env, '_stuck_count'):
                        env._stuck_count[eid] = 0
                    if hasattr(env, '_recovery_timer'):
                        env._recovery_timer[eid] = 0

            # Dock timeout
            elif dock_step[eid].item() > DOCK_TIMEOUT:
                phase[eid] = 3
                n_attempts += 1
                n_dock_timeout += 1

            # Box fell
            elif box_z_l < 0.30 and contain < 150:
                phase[eid] = 3
                n_attempts += 1
                n_box_fell += 1

        elif phase[eid] == 1:  # climb
            # Wait for drone to reach target altitude with box
            drone_z_l = cur_z - origin_z

            if drone_z_l > CLIMB_TARGET_Z and bd < 0.30:
                # SUCCESS: capture state
                capture_state(eid)
                n_captured += 1
                n_attempts += 1
                phase[eid] = 3
            elif bd > 0.50:
                # Box lost during climb
                phase[eid] = 3
                n_attempts += 1
                n_climb_fail += 1
            elif dock_step[eid].item() > DOCK_TIMEOUT + 1500:
                # Climb timeout
                phase[eid] = 3
                n_attempts += 1
                n_climb_fail += 1

    # Recycle done envs
    done_mask = (phase == 3)
    if done_mask.any() and n_captured < TARGET_STATES:
        done_ids = done_mask.nonzero(as_tuple=False).view(-1)
        setup_dock(done_ids)
        for eid in done_ids.tolist():
            origin = env.scene.env_origins[eid]
            bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
            obj_state = torch.zeros(1, 13, device=device)
            obj_state[0, :3] = bp
            obj_state[0, 3] = 1.0
            env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
            env.object_pos[eid] = bp

    if step % 5000 == 0:
        print(f"  [step {step:>6}] captured={n_captured}/{TARGET_STATES}")

# ============================================================
# Save
# ============================================================
os.makedirs("data", exist_ok=True)
save_dict = {}
for key, val_list in captured.items():
    if len(val_list) > 0:
        save_dict[key] = torch.stack(val_list[:TARGET_STATES])
save_dict['n_states'] = torch.tensor([len(save_dict['drone_pos_local'])])

save_path = "data/grasp_states.pt"
torch.save(save_dict, save_path)
print(f"\n{'='*60}")
print(f"  GRASP STATE GENERATION RESULTS")
print(f"{'='*60}")
print(f"  Total attempts:    {n_attempts}")
print(f"  Success (captured):{n_captured}/{n_attempts} ({100*n_captured/max(n_attempts,1):.1f}%)")
print(f"  Dock timeout:      {n_dock_timeout}/{n_attempts} ({100*n_dock_timeout/max(n_attempts,1):.1f}%)")
print(f"  Box fell:          {n_box_fell}/{n_attempts} ({100*n_box_fell/max(n_attempts,1):.1f}%)")
print(f"  Climb fail:        {n_climb_fail}/{n_attempts} ({100*n_climb_fail/max(n_attempts,1):.1f}%)")
print(f"  False grasps:      {n_false_grasp} (retried, not counted as attempt)")
print(f"")
print(f"  SAVED {save_dict['n_states'].item()} states to {save_path}")
print(f"  Drone z range: {save_dict['drone_pos_local'][:, 2].min():.2f}~{save_dict['drone_pos_local'][:, 2].max():.2f}")
print(f"  Box-drone XY offset: mean={torch.norm(save_dict['box_pos_local'][:, :2] - save_dict['drone_pos_local'][:, :2], dim=-1).mean():.4f}")
print(f"  Drone vel_z: mean={save_dict['drone_vel'][:, 2].mean():.3f} std={save_dict['drone_vel'][:, 2].std():.3f}")
print(f"{'='*60}")

env.close()
simulation_app.close()
