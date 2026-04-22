"""
Simplified waypoint-following drone environment.

19D obs, 4D action. Uses same AttitudeController + MotorModel.
Separate from GripperDroneEnv — no grasping, no object tracking.
Designed for Stage 1 (approach flight) and Stage 4 (loaded flight).
"""
from __future__ import annotations

import math
import torch
from isaaclab.envs import DirectRLEnv

from envs.waypoint_cfg import WaypointEnvCfg
from controllers.drone_ctrl import AttitudeController, quat_to_rot_matrix


class WaypointDroneEnv(DirectRLEnv):
    cfg: WaypointEnvCfg

    def __init__(self, cfg: WaypointEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Robot reference
        self.robot = self.scene["robot"]

        # Motor body IDs (from URDF: motor_0..3)
        motor_names = ["motor_0", "motor_1", "motor_2", "motor_3"]
        self.motor_body_ids = [self.robot.find_bodies(n)[0][0] for n in motor_names]
        self.base_body_id = self.robot.find_bodies("base_link")[0]

        # Gripper plates
        self.plate_joint_ids = self.robot.find_joints("plate_joint.*")[0]

        # Grasp box (loaded mode only)
        self.grasp_box = self.scene["grasp_box"] if self.cfg.mode == "loaded" else None
        self.contain_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Attitude controller (same as GripperDroneEnv)
        self.attitude_ctrl = AttitudeController(
            num_envs=self.num_envs,
            device=self.device,
        )

        # External forces/torques buffer
        num_bodies = self.robot.num_bodies
        self.ext_forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self.ext_torques = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)

        # Action buffers
        self.raw_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Goal position
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Step counter
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Domain randomization
        self.mass_scale = torch.ones(self.num_envs, device=self.device)
        self.motor_kf_scale = torch.ones(self.num_envs, device=self.device)
        self.payload_mass = torch.zeros(self.num_envs, device=self.device)

        # Wind
        self.wind_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.wind_phase = torch.zeros(self.num_envs, 3, device=self.device)

        # Action scaling: ±8 (full authority for recovery from disturbances)
        self.action_low = torch.tensor([-8.0, -8.0, -8.0, -math.pi], device=self.device)
        self.action_high = torch.tensor([8.0, 8.0, 8.0, math.pi], device=self.device)

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + (raw_action + 1.0) / 2.0 * (self.action_high - self.action_low)

    # ========================================================================
    # Scene setup
    # ========================================================================

    def _setup_scene(self):
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ========================================================================
    # Pre-physics: process actions
    # ========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_action = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        scaled = self._scale_action(actions)

        # Store for attitude controller
        self._accel_cmd = scaled[:, :3]
        self._yaw_ref = scaled[:, 3]

        # Gripper: closed for loaded (hold box), open for flight
        if self.cfg.mode == "loaded":
            plate_targets = torch.full((self.num_envs, 2), -0.087, device=self.device)
        else:
            plate_targets = torch.full((self.num_envs, 2), 0.873, device=self.device)
        self.robot.set_joint_position_target(plate_targets, joint_ids=self.plate_joint_ids)

    def _apply_action(self):
        """Apply forces/torques via attitude controller. Called at 300Hz."""
        quat_w = self.robot.data.root_quat_w
        ang_vel_b = self.robot.data.root_ang_vel_b

        # Rate command = 0 (attitude controller derives from accel)
        rate_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        forces_b, torques_b = self.attitude_ctrl.compute(
            accel_cmd_b=self._accel_cmd,
            rate_cmd_b=rate_cmd,
            yaw_ref=self._yaw_ref,
            quat_w=quat_w,
            ang_vel_b=ang_vel_b,
            dt=self.cfg.sim.dt,
        )

        self.ext_forces.zero_()
        self.ext_torques.zero_()

        # Motor thrust scaling (domain randomization)
        kf_scale = self.motor_kf_scale.unsqueeze(-1).unsqueeze(-1)
        forces_b = forces_b * kf_scale
        torques_b = torques_b * kf_scale

        for i in range(4):
            motor_id = self.motor_body_ids[i]
            self.ext_forces[:, motor_id, :] = forces_b[:, i, :]
            self.ext_torques[:, motor_id, :] = torques_b[:, i, :]

        # World-frame disturbances
        world_force = torch.zeros(self.num_envs, 3, device=self.device)
        nominal_mass = 1.080
        world_force[:, 2] -= (self.mass_scale - 1.0) * nominal_mass * 9.81

        # Payload weight: physical box applies its own gravity through contact
        # No virtual force needed — PhysX handles it

        # Wind
        self._update_wind()
        world_force += self.wind_force

        R = quat_to_rot_matrix(quat_w)
        body_force = torch.bmm(R.transpose(1, 2), world_force.unsqueeze(-1)).squeeze(-1)
        self.ext_forces[:, self.base_body_id[0], :] += body_force

        self.robot.set_external_force_and_torque(self.ext_forces, self.ext_torques)

    def _update_wind(self):
        dr = self.cfg.domain_rand
        t = self.step_count.float() * self.cfg.sim.dt
        for axis in range(3):
            self.wind_force[:, axis] = (
                dr.wind_force_std
                * torch.sin(2 * math.pi * dr.wind_freq * t + self.wind_phase[:, axis])
            )

    # ========================================================================
    # Observations (19D)
    # ========================================================================

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """
        22D obs (flight) or 23D (loaded):
          vel_b(3), ang_vel_b(3), R_flat(9), goal_b(3), prev_action(4)
          + payload_est(1) for loaded mode
        """
        pos_w = self.robot.data.root_pos_w
        vel_b = self.robot.data.root_lin_vel_b
        ang_vel_b = self.robot.data.root_ang_vel_b
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        # Goal in body frame
        goal_err_w = self.goal_pos - pos_w
        goal_b = torch.bmm(R.transpose(1, 2), goal_err_w.unsqueeze(-1)).squeeze(-1)

        # Rotation matrix flattened
        R_flat = R.reshape(self.num_envs, 9)

        # Previous action (full 4D, normalized to ~[-1,1])
        prev_action_norm = self.prev_action.clone()
        prev_action_norm[:, :3] /= 8.0    # accel range [-8,8] → [-1,1]
        prev_action_norm[:, 3] /= math.pi  # yaw range [-π,π] → [-1,1]

        # Add noise
        dr = self.cfg.domain_rand
        vel_b_noisy = vel_b + dr.vel_noise_std * torch.randn_like(vel_b)
        goal_b_noisy = goal_b + dr.pos_noise_std * torch.randn_like(goal_b)

        obs_parts = [
            vel_b_noisy,       # (3)
            ang_vel_b,         # (3)
            R_flat,            # (9)
            goal_b_noisy,      # (3)
            prev_action_norm,  # (4)
        ]  # Total: 22D

        # Stage 4: add payload estimate
        if self.cfg.mode == "loaded":
            payload_est = self.payload_mass.unsqueeze(-1)
            payload_est = payload_est + 0.02 * torch.randn_like(payload_est)  # noise
            obs_parts.append(payload_est)  # +1D = 23D

        obs = torch.cat(obs_parts, dim=-1)

        self.step_count += 1
        return {"policy": obs}

    # ========================================================================
    # Rewards
    # ========================================================================

    def _get_rewards(self) -> torch.Tensor:
        pos_w = self.robot.data.root_pos_w
        ang_vel_b = self.robot.data.root_ang_vel_b
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        # dt scaling (Genesis multiplies ALL rewards by sim dt)
        dt = self.cfg.sim.dt * self.cfg.decimation  # policy dt

        rel_pos = self.goal_pos - pos_w
        dist_sq = torch.sum(rel_pos ** 2, dim=-1)

        # === REWARD: efficient direct flight (NO potential reward — prevents diving exploit) ===

        # Direction: velocity projected onto goal direction (ONLY path-dependent reward)
        vel_w = self.robot.data.root_lin_vel_w
        pos_err = torch.sqrt(dist_sq).clamp(min=0.1)
        goal_dir = rel_pos / pos_err.unsqueeze(-1)        # unit direction to goal
        vel_toward_goal = torch.sum(vel_w * goal_dir, dim=-1)  # m/s toward goal
        r_direction = 0.5 * vel_toward_goal.clamp(-3.0, 3.0)  # cap at 3m/s (no reckless speed bonus)

        # Arrival: time-based bonus (faster = more reward)
        if not hasattr(self, '_steps_since_goal'):
            self._steps_since_goal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_goal += 1
        arrived = (pos_err < 0.3).float()
        time_sec = self._steps_since_goal.float() / 150.0
        r_arrive = arrived * 10.0 / (time_sec + 0.5)  # 0.5s→10, 2s→4, 5s→1.8

        # Crash penalty
        crashed = (pos_w[:, 2] < self.cfg.min_altitude).float()
        r_crash = -5.0 * crashed

        # Penalties (no dt — must be felt relative to r_direction ~0.5-1.5/step)
        action = self.raw_actions
        prev = self.prev_action
        r_smooth = -0.01 * torch.sum((action - prev) ** 2, dim=-1)  # jerky flight penalty
        ang_vel_norm = torch.norm(ang_vel_b, dim=-1)
        r_angular = -0.02 * ang_vel_norm  # spin penalty

        # Tilt penalty: stable flight for both modes
        tilt_angle = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
        if self.cfg.mode == "loaded":
            # Loaded: strict (20° threshold, heavy penalty — box pendulum)
            tilt_excess = (tilt_angle - 0.35).clamp(min=0)  # 0.35 rad ≈ 20°
            r_tilt = -5.0 * tilt_excess
        else:
            # Flight: moderate (30° threshold, lighter penalty — no payload)
            tilt_excess = (tilt_angle - 0.52).clamp(min=0)  # 0.52 rad ≈ 30°
            r_tilt = -3.0 * tilt_excess

        rewards = r_direction + r_arrive + r_crash + r_smooth + r_angular + r_tilt

        # Goal regeneration: reached (30cm) OR timeout (3 seconds)
        reached = (pos_err < 0.3)
        timed_out = (self._steps_since_goal > 450)  # 3s at 150Hz
        regen_mask = reached | timed_out

        # Timeout penalty (failed to reach goal in time)
        r_timeout = -2.0 * timed_out.float()
        rewards = rewards + r_timeout

        if regen_mask.any():
            regen_ids = regen_mask.nonzero(as_tuple=False).view(-1)
            n = len(regen_ids)
            new_xy = (torch.rand(n, 2, device=self.device) * 2 - 1) * self.cfg.goal_range_xy
            z_lo, z_hi = self.cfg.goal_range_z
            new_z = z_lo + torch.rand(n, 1, device=self.device) * (z_hi - z_lo)
            self.goal_pos[regen_ids] = torch.cat([new_xy, new_z], dim=-1)
            self._steps_since_goal[regen_ids] = 0

        return rewards

    # ========================================================================
    # Termination
    # ========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.robot.data.root_pos_w
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        tilt = 1.0 - R[:, 2, 2]
        max_tilt = 1.0 - math.cos(math.radians(self.cfg.max_tilt_deg))

        too_low = pos_w[:, 2] < self.cfg.min_altitude
        too_far = torch.norm(pos_w[:, :2], dim=-1) > self.cfg.max_distance
        too_tilted = tilt > max_tilt

        terminated = too_low | too_far | too_tilted

        max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        truncated = self.step_count >= max_steps

        return terminated, truncated

    # ========================================================================
    # Reset
    # ========================================================================

    def _reset_idx(self, env_ids: torch.Tensor):
        num_reset = len(env_ids)

        # Domain randomization
        dr = self.cfg.domain_rand
        self.mass_scale[env_ids] = (
            dr.mass_scale_range[0]
            + torch.rand(num_reset, device=self.device)
            * (dr.mass_scale_range[1] - dr.mass_scale_range[0])
        )
        self.motor_kf_scale[env_ids] = (
            dr.motor_k_f_scale[0]
            + torch.rand(num_reset, device=self.device)
            * (dr.motor_k_f_scale[1] - dr.motor_k_f_scale[0])
        )

        # Payload (Stage 4 only)
        if self.cfg.mode == "loaded":
            self.payload_mass[env_ids] = (
                self.cfg.payload_mass_range[0]
                + torch.rand(num_reset, device=self.device)
                * (self.cfg.payload_mass_range[1] - self.cfg.payload_mass_range[0])
            )
            # Update controller mass
            drone_mass = self.attitude_ctrl.base_mass * self.mass_scale[env_ids]
            self.attitude_ctrl.mass[env_ids] = drone_mass + self.payload_mass[env_ids]
            # Gripper stays closed
            self.contain_hold_count[env_ids] = 200
        else:
            self.attitude_ctrl.mass[env_ids] = (
                self.attitude_ctrl.base_mass * self.mass_scale[env_ids]
            )

        # Wind phase
        self.wind_phase[env_ids] = torch.rand(num_reset, 3, device=self.device) * 2 * math.pi

        # Reset actions
        self.raw_actions[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.step_count[env_ids] = 0
        if hasattr(self, '_steps_since_goal'):
            self._steps_since_goal[env_ids] = 0

        # Reset attitude controller
        self.attitude_ctrl.reset(env_ids)

        # Spawn drone
        spawn_offset = self.cfg.spawn_spread * (
            torch.rand(num_reset, 3, device=self.device) - 0.5
        )
        spawn_offset[:, 2] *= 0.5  # less Z variation

        default_pos = torch.tensor([0.0, 0.0, self.cfg.spawn_z], device=self.device)
        root_pos = default_pos + spawn_offset

        # Random small orientation perturbation
        small_euler = 0.05 * (torch.rand(num_reset, 3, device=self.device) - 0.5)
        root_quat = _euler_to_quat(small_euler)

        root_state = torch.zeros(num_reset, 13, device=self.device)
        root_state[:, :3] = root_pos
        root_state[:, 3:7] = root_quat
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # Loaded mode: place box in gripper + close plates
        if self.cfg.mode == "loaded" and self.grasp_box is not None:
            # Box at gripper center (8cm below drone CoM)
            R = quat_to_rot_matrix(root_quat)
            gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=self.device)
            box_pos = root_pos + torch.bmm(
                R, gripper_offset.expand(num_reset, 3).unsqueeze(-1)
            ).squeeze(-1)
            box_state = torch.zeros(num_reset, 13, device=self.device)
            box_state[:, :3] = box_pos
            box_state[:, 3:7] = root_quat  # same orientation as drone
            self.grasp_box.write_root_state_to_sim(box_state, env_ids)

            # Close gripper plates
            joint_pos = self.robot.data.joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            for jid in self.plate_joint_ids:
                joint_pos[:, jid] = -0.087  # closed
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            closed_target = torch.full((num_reset, 2), -0.087, device=self.device)
            self.robot.set_joint_position_target(closed_target, joint_ids=self.plate_joint_ids, env_ids=env_ids)

        # Random goal
        goal_xy = (torch.rand(num_reset, 2, device=self.device) * 2 - 1) * self.cfg.goal_range_xy
        goal_z_lo, goal_z_hi = self.cfg.goal_range_z
        goal_z = goal_z_lo + torch.rand(num_reset, 1, device=self.device) * (goal_z_hi - goal_z_lo)
        self.goal_pos[env_ids] = torch.cat([goal_xy, goal_z], dim=-1)


def _euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    r, p, y = euler[:, 0], euler[:, 1], euler[:, 2]
    cr, sr = torch.cos(r / 2), torch.sin(r / 2)
    cp, sp = torch.cos(p / 2), torch.sin(p / 2)
    cy, sy = torch.cos(y / 2), torch.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_q = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y_q, z], dim=-1)
