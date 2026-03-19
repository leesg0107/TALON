"""
Gripper-Drone Isaac Lab environment.

DirectRLEnv implementation supporting all 5 training stages.
Flight physics use external force/torque application (no propeller joints needed).
Inner-loop attitude controller + motor model run at simulation rate (300Hz).
RL policy runs at control rate (150Hz, via decimation=2).
"""

from __future__ import annotations

import torch
import math
from typing import Any

from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.utils.math as math_utils

from .env_cfg import GripperDroneEnvCfg, Stage
from controllers.drone_ctrl import AttitudeController, quat_to_rot_matrix
from rewards.reward_fn import (
    compute_stage1_rewards,
    compute_stage2_rewards,
    compute_stage3_rewards,
    compute_stage4_rewards,
    compute_stage5_rewards,
)


class GripperDroneEnv(DirectRLEnv):
    """Gripper-Drone training environment.

    Supports curriculum stages 1-5 via configuration. Each stage adds
    observations, actions, and reward terms incrementally.
    """

    cfg: GripperDroneEnvCfg

    def __init__(self, cfg: GripperDroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache body indices for force application
        self.motor_body_ids, _ = self.robot.find_bodies("motor_.*")
        self.base_body_id, _ = self.robot.find_bodies("base_link")
        self.plate_left_id, _ = self.robot.find_bodies("plate_left")
        self.plate_right_id, _ = self.robot.find_bodies("plate_right")

        # Joint indices for gripper plates
        self.plate_joint_ids, _ = self.robot.find_joints("plate_joint_.*")

        # Number of bodies for force buffers
        self.num_bodies = self.robot.num_bodies

        # --- Inner-loop controller ---
        self.attitude_ctrl = AttitudeController(
            num_envs=self.num_envs,
            device=self.device,
        )

        # --- State buffers ---
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # External force/torque buffers (applied to all bodies)
        self.ext_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.ext_torques = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)

        # Wind disturbance (slowly varying)
        self.wind_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.wind_phase = torch.zeros(self.num_envs, 3, device=self.device)

        # Domain randomization samples (per episode)
        self.mass_scale = torch.ones(self.num_envs, device=self.device)
        self.payload_mass = torch.zeros(self.num_envs, device=self.device)
        self.motor_kf_scale = torch.ones(self.num_envs, device=self.device)

        # Observation normalization (running stats)
        self.obs_mean = torch.zeros(self.cfg.num_observations, device=self.device)
        self.obs_var = torch.ones(self.cfg.num_observations, device=self.device)
        self.obs_count = 1e-4

        # --- Stage 3+ state ---
        if self.cfg.stage.value >= Stage.GRASPING.value:
            self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self.object_mass = torch.zeros(self.num_envs, device=self.device)
            self.is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # --- Action scaling ---
        self._setup_action_scaling()

    def _setup_action_scaling(self):
        """Define action space scaling from [-1, 1] to physical units."""
        # [ax, ay, az]: [-8, 8] m/s^2
        # [wx, wy, wz]: [-3, 3] rad/s
        # [yaw_ref]:    [-pi, pi] rad
        # [gripper]:    [-5deg, 50deg] = [-0.087, 0.873] rad

        self.action_low = torch.tensor(
            [-8.0, -8.0, -8.0, -3.0, -3.0, -3.0, -math.pi],
            device=self.device,
        )
        self.action_high = torch.tensor(
            [8.0, 8.0, 8.0, 3.0, 3.0, 3.0, math.pi],
            device=self.device,
        )

        if self.cfg.num_actions == 8:
            self.action_low = torch.cat([
                self.action_low,
                torch.tensor([-0.087266], device=self.device),
            ])
            self.action_high = torch.cat([
                self.action_high,
                torch.tensor([0.872665], device=self.device),
            ])

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Scale raw [-1, 1] policy output to physical action space."""
        return self.action_low + (raw_action + 1.0) / 2.0 * (self.action_high - self.action_low)

    # ========================================================================
    # Scene setup
    # ========================================================================

    def _setup_scene(self):
        """Add robot and ground to the scene."""
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot

        # Ground plane is auto-spawned from scene config
        self.cfg.scene.ground.func(
            "/World/ground",
            self.cfg.scene.ground,
        )

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions (optional: disable self-collision between arms and body)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ========================================================================
    # Pre-physics: process RL actions through inner-loop controller
    # ========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process RL actions. Called once per RL step (150 Hz)."""
        self.raw_actions = actions.clone()
        self.scaled_actions = self._scale_action(actions)

        # Gripper plate commands (stage 3+)
        if self.cfg.num_actions == 8:
            gripper_cmd = self.scaled_actions[:, 7]
            # Apply to both plates (symmetric)
            plate_targets = torch.stack([gripper_cmd, gripper_cmd], dim=-1)
            self.robot.set_joint_position_target(plate_targets, joint_ids=self.plate_joint_ids)
        else:
            # Lock plates at 45 deg (landing config)
            landing_angle = 0.785398  # 45 deg
            plate_targets = torch.full(
                (self.num_envs, 2), landing_angle, device=self.device,
            )
            self.robot.set_joint_position_target(plate_targets, joint_ids=self.plate_joint_ids)

    def _apply_action(self):
        """Apply forces/torques to robot. Called every simulation step (300 Hz)."""
        # Extract flight commands
        accel_cmd_b = self.scaled_actions[:, :3]
        rate_cmd_b = self.scaled_actions[:, 3:6]
        yaw_ref = self.scaled_actions[:, 6]

        # Get current state
        quat_w = self.robot.data.root_quat_w          # (N, 4) wxyz
        ang_vel_b = self.robot.data.root_ang_vel_b     # (N, 3)

        # Run inner-loop attitude controller
        forces_b, torques_b = self.attitude_ctrl.compute(
            accel_cmd_b=accel_cmd_b,
            rate_cmd_b=rate_cmd_b,
            yaw_ref=yaw_ref,
            quat_w=quat_w,
            ang_vel_b=ang_vel_b,
            dt=self.cfg.sim.dt,
        )

        # Transform motor forces from body frame to world frame
        R = quat_to_rot_matrix(quat_w)  # (N, 3, 3)

        self.ext_forces.zero_()
        self.ext_torques.zero_()

        for i in range(4):
            motor_id = self.motor_body_ids[i]
            # Transform force to world frame
            f_world = torch.bmm(R, forces_b[:, i:i+1, :].transpose(1, 2)).squeeze(-1)
            t_world = torch.bmm(R, torques_b[:, i:i+1, :].transpose(1, 2)).squeeze(-1)
            self.ext_forces[:, motor_id, :] = f_world
            self.ext_torques[:, motor_id, :] = t_world

        # Add wind disturbance to base_link
        self._update_wind()
        self.ext_forces[:, self.base_body_id[0], :] += self.wind_force

        # Apply payload weight if grasped (stage 3+)
        if self.cfg.stage.value >= Stage.GRASPING.value:
            payload_weight = self.payload_mass * 9.81
            self.ext_forces[:, self.base_body_id[0], 2] -= payload_weight * self.is_grasped.float()

        # Apply to simulation
        self.robot.set_external_force_and_torque(self.ext_forces, self.ext_torques)

    def _update_wind(self):
        """Update slowly-varying wind disturbance."""
        dt = self.cfg.sim.dt
        freq = self.cfg.domain_rand.wind_freq
        std = self.cfg.domain_rand.wind_force_std

        self.wind_phase += 2.0 * math.pi * freq * dt
        # Slowly varying sinusoidal wind + noise
        self.wind_force = std * torch.sin(self.wind_phase)

    # ========================================================================
    # Observations
    # ========================================================================

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observation tensor.

        Stage 1-2: 23D = [v_b(3), w_b(3), R(9), p_goal_b(3), p_err_w(3), prev_action_xy(2)]
        Stage 3-5: 31D = above + [theta_plates(2), p_obj_b(3), grasp_flag(1), payload_est(1), d_obj(1)]
        """
        # Base state
        pos_w = self.robot.data.root_pos_w                # (N, 3)
        quat_w = self.robot.data.root_quat_w              # (N, 4) wxyz
        vel_b = self.robot.data.root_lin_vel_b             # (N, 3)
        ang_vel_b = self.robot.data.root_ang_vel_b         # (N, 3)
        R = quat_to_rot_matrix(quat_w)                     # (N, 3, 3)

        # Goal in body frame: p_goal_b = R^T @ (p_goal - p_body)
        p_err_w = self.goal_pos - pos_w
        p_goal_b = torch.bmm(R.transpose(1, 2), p_err_w.unsqueeze(-1)).squeeze(-1)

        # Flatten rotation matrix to 9D
        R_flat = R.reshape(self.num_envs, 9)

        # Add sensor noise
        vel_b_noisy = vel_b + self.cfg.domain_rand.vel_noise_std * torch.randn_like(vel_b)
        p_err_w_noisy = p_err_w + self.cfg.domain_rand.pos_noise_std * torch.randn_like(p_err_w)
        p_goal_b_noisy = torch.bmm(
            R.transpose(1, 2),
            (p_err_w_noisy).unsqueeze(-1),
        ).squeeze(-1)

        obs_parts = [
            vel_b_noisy,            # (N, 3)
            ang_vel_b,              # (N, 3)
            R_flat,                 # (N, 9)
            p_goal_b_noisy,         # (N, 3)
            p_err_w_noisy,          # (N, 3)
            self.prev_action[:, :2],  # (N, 2) previous accel XY
        ]

        # Stage 3+ additional observations
        if self.cfg.stage.value >= Stage.GRASPING.value:
            # Plate joint angles, normalized to [-1, 1]
            joint_pos = self.robot.data.joint_pos[:, self.plate_joint_ids]  # (N, 2)
            # Normalize: range is [-0.087, 0.873], center at 0.393, half-range 0.48
            theta_norm = (joint_pos - 0.393) / 0.48

            # Object position in body frame
            p_obj_err = self.object_pos - pos_w
            p_obj_b = torch.bmm(R.transpose(1, 2), p_obj_err.unsqueeze(-1)).squeeze(-1)
            # Add detection noise
            p_obj_b += self.cfg.domain_rand.obj_detection_noise_std * torch.randn_like(p_obj_b)

            # Grasp flag
            grasp_flag = self.is_grasped.float().unsqueeze(-1)

            # Payload mass estimate (noisy)
            payload_est = (self.payload_mass + 0.02 * torch.randn(self.num_envs, device=self.device)).unsqueeze(-1)

            # Distance to object
            d_obj = torch.norm(p_obj_err, dim=-1, keepdim=True)

            obs_parts.extend([
                theta_norm,     # (N, 2)
                p_obj_b,        # (N, 3)
                grasp_flag,     # (N, 1)
                payload_est,    # (N, 1)
                d_obj,          # (N, 1)
            ])

        obs = torch.cat(obs_parts, dim=-1)

        # Running normalization
        self._update_obs_stats(obs)
        obs_normalized = (obs - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-5)

        self.prev_action = self.raw_actions.clone()
        self.step_count += 1

        return {"policy": obs_normalized}

    def _update_obs_stats(self, obs: torch.Tensor):
        """Update running mean/variance for observation normalization."""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.obs_count * batch_count / total_count
        new_var = m2 / total_count

        self.obs_mean = new_mean
        self.obs_var = new_var
        self.obs_count = total_count

    # ========================================================================
    # Rewards
    # ========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Compute stage-appropriate rewards."""
        pos_w = self.robot.data.root_pos_w
        vel_b = self.robot.data.root_lin_vel_b
        vel_w = self.robot.data.root_lin_vel_w
        ang_vel_b = self.robot.data.root_ang_vel_b
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        if self.cfg.stage == Stage.BASIC_FLIGHT:
            rewards, info = compute_stage1_rewards(
                pos_w=pos_w, vel_b=vel_b, rot_matrix=R,
                goal_w=self.goal_pos, action=self.raw_actions,
                prev_action=self.prev_action,
            )

        elif self.cfg.stage == Stage.PRECISION_APPROACH:
            target_vz = torch.full((self.num_envs,), self.cfg.descent_speed, device=self.device)
            rewards, info = compute_stage2_rewards(
                pos_w=pos_w, vel_w=vel_w, rot_matrix=R,
                goal_w=self.goal_pos, target_vz=target_vz,
                action=self.raw_actions, prev_action=self.prev_action,
            )

        elif self.cfg.stage == Stage.GRASPING:
            self._check_grasp_conditions(pos_w, vel_w, R)
            just_grasped = self.is_grasped & (~self.was_grasped)
            just_dropped = (~self.is_grasped) & self.was_grasped

            plate_angles = self.robot.data.joint_pos[:, self.plate_joint_ids[0]]
            plate_norm = (plate_angles - 0.393) / 0.48

            rewards, info = compute_stage3_rewards(
                pos_w=pos_w, vel_w=vel_w, rot_matrix=R,
                obj_pos_w=self.object_pos, plate_angle=plate_norm,
                is_grasped=self.is_grasped, was_grasped=self.was_grasped,
                just_grasped=just_grasped, just_dropped=just_dropped,
                action=self.raw_actions, prev_action=self.prev_action,
            )
            self.was_grasped = self.is_grasped.clone()

        elif self.cfg.stage == Stage.LOADED_FLIGHT:
            rewards, info = compute_stage4_rewards(
                pos_w=pos_w, vel_b=vel_b, ang_vel_b=ang_vel_b,
                rot_matrix=R, goal_w=self.goal_pos,
                is_grasped=self.is_grasped, action=self.raw_actions,
                prev_action=self.prev_action,
            )

        elif self.cfg.stage == Stage.RELEASE:
            just_released = self.was_grasped & (~self.is_grasped)
            plate_angles = self.robot.data.joint_pos[:, self.plate_joint_ids[0]]
            plate_norm = (plate_angles - 0.393) / 0.48

            rewards, info = compute_stage5_rewards(
                pos_w=pos_w, vel_w=vel_w, rot_matrix=R,
                delivery_pos_w=self.goal_pos, plate_angle=plate_norm,
                is_grasped=self.is_grasped, just_released=just_released,
                action=self.raw_actions, prev_action=self.prev_action,
            )
            self.was_grasped = self.is_grasped.clone()
        else:
            rewards = torch.zeros(self.num_envs, device=self.device)
            info = {}

        # Log reward components
        self.extras["log"] = {k: v.mean().item() for k, v in info.items()}

        return rewards

    def _check_grasp_conditions(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        R: torch.Tensor,
    ):
        """Check and update grasp state for Stage 3."""
        d_obj = torch.norm(pos_w - self.object_pos, dim=-1)
        speed = torch.norm(vel_w, dim=-1)
        tilt = 1.0 - R[:, 2, 2]

        trigger = (
            (d_obj < self.cfg.grasp_trigger_dist) &
            (speed < self.cfg.grasp_trigger_speed) &
            (tilt < self.cfg.grasp_trigger_tilt) &
            (~self.is_grasped)
        )

        if self.cfg.auto_grasp:
            # Auto-grasp with probability
            auto_mask = torch.rand(self.num_envs, device=self.device) < self.cfg.auto_grasp_prob
            self.is_grasped |= (trigger & auto_mask)
        else:
            # Learned grasp: check if plate angle < threshold (closing)
            plate_angle = self.robot.data.joint_pos[:, self.plate_joint_ids[0]]
            plates_closed = plate_angle < 0.05  # near 0 deg
            self.is_grasped |= (trigger & plates_closed)

        # Check for drop (object falls if plates open while grasped)
        if self.is_grasped.any():
            plate_angle = self.robot.data.joint_pos[:, self.plate_joint_ids[0]]
            excessive_tilt = tilt > 0.5  # ~45 deg
            plates_open = plate_angle > 0.3  # ~17 deg
            drop = self.is_grasped & (excessive_tilt | plates_open)
            self.is_grasped &= ~drop

    # ========================================================================
    # Termination
    # ========================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation flags."""
        pos_w = self.robot.data.root_pos_w
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        tilt = 1.0 - R[:, 2, 2]
        max_tilt = 1.0 - math.cos(math.radians(self.cfg.max_tilt_deg))

        # Termination conditions (bad states)
        too_low = pos_w[:, 2] < self.cfg.min_altitude
        too_far = torch.norm(pos_w[:, :2], dim=-1) > self.cfg.max_distance
        too_tilted = tilt > max_tilt

        terminated = too_low | too_far | too_tilted

        # Truncation (time limit)
        max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        truncated = self.step_count >= max_steps

        return terminated, truncated

    # ========================================================================
    # Reset
    # ========================================================================

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        num_reset = len(env_ids)

        # --- Domain randomization ---
        dr = self.cfg.domain_rand

        self.mass_scale[env_ids] = torch.empty(num_reset, device=self.device).uniform_(
            dr.mass_scale_range[0], dr.mass_scale_range[1]
        )
        self.payload_mass[env_ids] = torch.empty(num_reset, device=self.device).uniform_(
            dr.payload_mass_range[0], dr.payload_mass_range[1]
        )
        self.motor_kf_scale[env_ids] = torch.empty(num_reset, device=self.device).uniform_(
            dr.motor_k_f_scale[0], dr.motor_k_f_scale[1]
        )

        # Randomize wind phase
        self.wind_phase[env_ids] = torch.rand(num_reset, 3, device=self.device) * 2.0 * math.pi

        # --- Reset robot state ---
        # Random spawn position near (0, 0, 3)
        default_pos = torch.tensor([0.0, 0.0, 3.0], device=self.device)
        spawn_offset = self.cfg.spawn_spread * (torch.rand(num_reset, 3, device=self.device) - 0.5)
        spawn_offset[:, 2] *= 0.5  # Less Z variation
        root_pos = default_pos + spawn_offset

        # Identity orientation with small random perturbation
        small_euler = 0.05 * (torch.rand(num_reset, 3, device=self.device) - 0.5)
        root_quat = _euler_to_quat(small_euler)  # (N, 4) wxyz

        # Zero velocity
        root_vel = torch.zeros(num_reset, 6, device=self.device)

        # Write to simulation
        root_state = torch.cat([root_pos, root_quat, root_vel], dim=-1)  # (N, 13)
        self.robot.write_root_state_to_sim(root_state, env_ids)

        # Reset joints (plates at landing angle)
        joint_pos = torch.full(
            (num_reset, self.robot.num_joints), 0.0, device=self.device,
        )
        joint_vel = torch.zeros_like(joint_pos)
        # Set plate joints to 45 deg
        for jid in self.plate_joint_ids:
            joint_pos[:, jid] = 0.785398
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # --- Reset controller ---
        self.attitude_ctrl.reset(env_ids)

        # --- Reset task state ---
        self.step_count[env_ids] = 0
        self.prev_action[env_ids] = 0.0

        # Sample new goal
        self._sample_goals(env_ids)

        # Reset grasping state (stage 3+)
        if self.cfg.stage.value >= Stage.GRASPING.value:
            self.is_grasped[env_ids] = False
            self.was_grasped[env_ids] = False
            self._sample_objects(env_ids)

    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample goal positions based on current stage."""
        n = len(env_ids)

        if self.cfg.stage == Stage.BASIC_FLIGHT:
            xy = self.cfg.goal_pos_range_xy * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
            z_low, z_high = self.cfg.goal_pos_range_z
            z = torch.empty(n, 1, device=self.device).uniform_(z_low, z_high)
            self.goal_pos[env_ids] = torch.cat([xy, z], dim=-1)

        elif self.cfg.stage == Stage.PRECISION_APPROACH:
            # Goal starts at approach altitude, will descend over episode
            xy = self.cfg.goal_pos_range_xy * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
            z_low, z_high = self.cfg.target_z
            z = torch.empty(n, 1, device=self.device).uniform_(z_low, z_high)
            self.goal_pos[env_ids] = torch.cat([xy, z], dim=-1)

        elif self.cfg.stage in (Stage.GRASPING, Stage.RELEASE):
            # Goal is the object position (set in _sample_objects)
            pass

        elif self.cfg.stage == Stage.LOADED_FLIGHT:
            xy = self.cfg.delivery_range_xy * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
            z = torch.full((n, 1), self.cfg.delivery_z, device=self.device)
            self.goal_pos[env_ids] = torch.cat([xy, z], dim=-1)

    def _sample_objects(self, env_ids: torch.Tensor):
        """Sample object positions and properties for grasping stages."""
        n = len(env_ids)
        xy = self.cfg.goal_pos_range_xy * 0.5 * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
        z_low, z_high = self.cfg.target_z
        z = torch.empty(n, 1, device=self.device).uniform_(z_low, z_high)
        self.object_pos[env_ids] = torch.cat([xy, z], dim=-1)
        self.goal_pos[env_ids] = self.object_pos[env_ids].clone()

        m_low, m_high = self.cfg.object_mass_range
        self.object_mass[env_ids] = torch.empty(n, device=self.device).uniform_(m_low, m_high)


# ============================================================================
# Utility
# ============================================================================


def _euler_to_quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z).

    Args:
        euler: (N, 3) roll, pitch, yaw in radians

    Returns:
        quat: (N, 4) quaternion in [w, x, y, z] format
    """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)
