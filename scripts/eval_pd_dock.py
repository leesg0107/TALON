"""PD dock+climb evaluation — accurate grasp success measurement.

Spawns drone above box (matching end-to-end conditions),
runs PD dock → climb, verifies box actually follows drone.
Phases: dock(0) → climb_verify(1) → success/fail
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

NUM_ENVS = 128
DOCK_TIMEOUT = 1800   # 12s
CLIMB_TIMEOUT = 1500  # 10s

cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = NUM_ENVS
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 30.0
cfg.lock_gripper = True
cfg.residual_scale = 0.0
cfg.goal_pos_range_xy = 0.0
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

def _no_dones():
    return (torch.zeros(NUM_ENVS, dtype=torch.bool, device=device),
            torch.zeros(NUM_ENVS, dtype=torch.bool, device=device))
env._get_dones = _no_dones

# ============================================================
# Per-env state: phase 0=dock, 1=climb_verify, 2=done
# ============================================================
phase = torch.full((NUM_ENVS,), 2, dtype=torch.long, device=device)  # start as done → setup
dock_step = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
climb_step = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
max_contain = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)

results = {"total": 0, "full_grasp": 0, "dock_only": 0}
fail_reasons = {}
from collections import Counter

def record(reason):
    results["total"] += 1
    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

def setup_dock(eids):
    for eid in eids.tolist():
        origin = env.scene.env_origins[eid]
        box_pos = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)

        obj_state = torch.zeros(1, 13, device=device)
        obj_state[0, :3] = box_pos
        obj_state[0, 3] = 1.0
        env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
        env.object_pos[eid] = box_pos

        # Entry matching ACTUAL end-to-end dock entry logs (median values)
        xy_off = 0.08 * (2 * torch.rand(2, device=device) - 1)   # ±8cm (actual: mostly <10cm)
        z_above = 0.45 + 0.10 * torch.rand(1, device=device).item()  # 0.45-0.55m (actual: 0.4-0.55)
        vel_xy = 0.20 * (2 * torch.rand(2, device=device) - 1)   # ±0.2 m/s (actual: mostly <0.3)

        root_state = torch.zeros(1, 13, device=device)
        root_state[0, 0] = box_pos[0] + xy_off[0]
        root_state[0, 1] = box_pos[1] + xy_off[1]
        root_state[0, 2] = box_pos[2] + z_above
        root_state[0, 3] = 1.0
        root_state[0, 7:9] = vel_xy
        env.robot.write_root_state_to_sim(root_state, torch.tensor([eid], device=device))

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
        env.attitude_ctrl.reset(torch.tensor([eid], device=device))  # reset motor state

        phase[eid] = 3  # settle (wait 1 step for teleport to commit)
        dock_step[eid] = 0
        climb_step[eid] = 0
        max_contain[eid] = 0

    env.reset_buf[eids] = False
    env.reset_terminated[eids] = False
    env.reset_time_outs[eids] = False

# ============================================================
# Init
# ============================================================
obs, _ = env_wrapped.reset()
all_ids = torch.arange(NUM_ENVS, device=device)
setup_dock(all_ids)

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
# phase already set to 3 (settle) by setup_dock — will transition to 0 (dock) on first step

TARGET = 200
print(f"\n{'='*60}")
print(f"  PD DOCK+CLIMB EVAL ({NUM_ENVS} envs, target={TARGET})")
print(f"  Entry: ±8cm XY, 0.45-0.55m above, ±0.2 m/s (actual e2e dock entry)")
print(f"{'='*60}\n")

# ============================================================
# Main loop
# ============================================================
for step in range(200000):
    if results["total"] >= TARGET:
        break

    pos_w = env.robot.data.root_pos_w
    obj_pos = env.object_pos

    # ---- SETTLE PHASE (3): wait for teleport to commit ----
    s_mask = (phase == 3)
    if s_mask.any():
        s_ids = s_mask.nonzero(as_tuple=False).view(-1)
        # Re-place box during settle
        for eid in s_ids.tolist():
            origin = env.scene.env_origins[eid]
            bp = torch.tensor([origin[0].item(), origin[1].item(), origin[2].item() + 0.54], device=device)
            obj_state = torch.zeros(1, 13, device=device)
            obj_state[0, :3] = bp
            obj_state[0, 3] = 1.0
            env.grasp_object.write_root_state_to_sim(obj_state, torch.tensor([eid], device=device))
            env.object_pos[eid] = bp
            env.contain_hold_count[eid] = 0
        phase[s_ids] = 0  # → dock on next step

    # --- Per-step diagnostic for env 0 (every 150 steps = 1s) ---
    if step % 150 == 0 and phase[0].item() == 0 and dock_step[0].item() > 0:
        eid = 0
        R_d = quat_to_rot_matrix(env.robot.data.root_quat_w[eid:eid+1])
        g_off = torch.tensor([0.0, 0.0, -0.08], device=device)
        g_pos = env.robot.data.root_pos_w[eid] + (R_d[0] @ g_off)
        o_pos = env.grasp_object.data.root_pos_w[eid]
        xy_d = torch.norm(g_pos[:2] - o_pos[:2]).item()
        alt_d = (g_pos[2] - (o_pos[2] + 0.04)).item()
        standoff_d = (g_pos[2] - o_pos[2]).item()
        vel_d = env.robot.data.root_lin_vel_w[eid].tolist()
        sa = env.scaled_actions[eid, :3].tolist()
        cc = env.contain_hold_count[eid].item()
        rt = env._recovery_timer[eid].item() if hasattr(env, '_recovery_timer') else 0
        print(f"  [env0 t={dock_step[0].item()/150:.1f}s] xy={xy_d:.3f} alt={alt_d:.3f} "
              f"standoff={standoff_d:.3f} vel_z={vel_d[2]:.3f} "
              f"accel_b_z={sa[2]:.3f} contain={cc} recovery={rt} "
              f"obj_z={o_pos[2].item():.3f} grip_z={g_pos[2].item():.3f}")

    env.bypass_analytical = False
    env_wrapped.step(torch.zeros(NUM_ENVS, 8, device=device))

    # ---- DOCK PHASE (0) ----
    d_mask = (phase == 0)
    if d_mask.any():
        d_ids = d_mask.nonzero(as_tuple=False).view(-1)
        dock_step[d_ids] += 1

        for eid in d_ids.tolist():
            contain = env.contain_hold_count[eid].item()
            max_contain[eid] = max(max_contain[eid].item(), contain)
            box_z_l = obj_pos[eid, 2].item() - env.scene.env_origins[eid, 2].item()

            # Box fell
            if box_z_l < 0.30 and contain < 150:
                phase[eid] = 2
                record("box_fell")
                continue

            # Contain >= 250: gripper closed (150) + dwell (100 steps) → verify climb
            if contain >= 325:
                bd = torch.norm(pos_w[eid] - obj_pos[eid]).item()
                if bd < 0.30:
                    # Transition to climb verify
                    drone_mass = env.attitude_ctrl.base_mass * env.mass_scale[eid]
                    env.attitude_ctrl.mass[eid] = drone_mass + 0.2
                    phase[eid] = 1  # climb_verify
                    climb_step[eid] = 0
                    results["dock_only"] += 1
                else:
                    # False grasp → retry
                    env.contain_hold_count[eid] = 0
                    if hasattr(env, '_stuck_count'):
                        env._stuck_count[eid] = 0
                    if hasattr(env, '_recovery_timer'):
                        env._recovery_timer[eid] = 0

            # Timeout
            elif dock_step[eid].item() > DOCK_TIMEOUT:
                phase[eid] = 2
                xy = torch.norm(pos_w[eid, :2] - obj_pos[eid, :2]).item()
                za = pos_w[eid, 2].item() - obj_pos[eid, 2].item()
                mc = max_contain[eid].item()
                cc = env.contain_hold_count[eid].item()
                # Gripper-centric diagnostics
                R_t = quat_to_rot_matrix(env.robot.data.root_quat_w[eid:eid+1])
                g_off = torch.tensor([0.0, 0.0, -0.08], device=device)
                g_pos = pos_w[eid] + (R_t[0] @ g_off)
                alt_a = (g_pos[2] - (obj_pos[eid, 2] + 0.04)).item()
                # box_local for overlap check
                box_off_w = obj_pos[eid] - g_pos
                box_local = (R_t[0].T @ box_off_w).tolist()

                if mc >= 150:
                    record(f"timeout_gripper_closed")
                elif za > 0.50:
                    record(f"timeout_high")
                elif xy > 0.15:
                    record(f"timeout_xy")
                else:
                    record(f"timeout_stuck")
                    print(f"    STUCK env{eid}: xy={xy:.3f} alt_above={alt_a:.3f} "
                          f"contain={cc} max_c={mc} "
                          f"box_local=[{box_local[0]:.3f},{box_local[1]:.3f},{box_local[2]:.3f}] "
                          f"standoff={g_pos[2].item()-obj_pos[eid,2].item():.3f}")

    # ---- CLIMB VERIFY (1): check box actually follows drone to 1.0m ----
    c_mask = (phase == 1)
    if c_mask.any():
        c_ids = c_mask.nonzero(as_tuple=False).view(-1)
        climb_step[c_ids] += 1

        for eid in c_ids.tolist():
            cur_z = pos_w[eid, 2].item()
            box_z = obj_pos[eid, 2].item()
            bd = torch.norm(pos_w[eid] - obj_pos[eid]).item()
            box_z_l = box_z - env.scene.env_origins[eid, 2].item()

            if cur_z > 1.0 and bd < 0.30:
                # Box followed drone to 1.0m → REAL grasp success
                phase[eid] = 2
                results["total"] += 1
                results["full_grasp"] += 1
                fail_reasons["full_grasp"] = fail_reasons.get("full_grasp", 0) + 1
            elif climb_step[eid].item() > CLIMB_TIMEOUT:
                phase[eid] = 2
                record("climb_timeout")
            elif box_z_l < 0.30:
                # Box fell during climb
                phase[eid] = 2
                record("box_fell_during_climb")
            elif bd > 0.50:
                phase[eid] = 2
                record("box_lost_during_climb")
            elif cur_z < 0.05:
                phase[eid] = 2
                record("crashed_during_climb")

    # ---- Recycle done envs ----
    done_mask = (phase == 2)
    if done_mask.any() and results["total"] < TARGET:
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
        n = max(results["total"], 1)
        print(f"  [step {step:>6}] total={results['total']}/{TARGET} "
              f"full_grasp={results['full_grasp']} ({100*results['full_grasp']/n:.0f}%) "
              f"dock_only={results['dock_only']}")

# ============================================================
# Results
# ============================================================
n = max(results["total"], 1)
print(f"\n{'='*60}")
print(f"  PD DOCK+CLIMB RESULTS (verified grasp)")
print(f"{'='*60}")
print(f"  Total attempts:     {results['total']}")
print(f"  Dock achieved:      {results['dock_only']}/{n} ({100*results['dock_only']/n:.1f}%)")
print(f"  Full grasp (climb): {results['full_grasp']}/{n} ({100*results['full_grasp']/n:.1f}%)")
print(f"  False grasp rate:   {results['dock_only']-results['full_grasp']}/{results['dock_only']} "
      f"({100*(results['dock_only']-results['full_grasp'])/max(results['dock_only'],1):.1f}%)")
print()
print(f"  --- Failure breakdown ---")
for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
    if reason != "full_grasp":
        print(f"  {reason:30s} {count:>4}/{n} ({100*count/n:.1f}%)")
print(f"{'='*60}")

env.close()
simulation_app.close()
