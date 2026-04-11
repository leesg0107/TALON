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

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import CUBOID_MARKER_CFG
import isaaclab.utils.math as math_utils

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

        # --- Runtime validation: ensure fixed joints were NOT merged ---
        assert len(self.motor_body_ids) == 4, (
            f"Expected 4 motor bodies, found {len(self.motor_body_ids)}. "
            f"Ensure merge_fixed_joints=False in UrdfFileCfg."
        )
        assert len(self.base_body_id) == 1, (
            f"Expected 1 base_link body, found {len(self.base_body_id)}."
        )

        # Joint indices for gripper plates
        self.plate_joint_ids, _ = self.robot.find_joints("plate_joint_.*")

        # Reference to grasp object (always present in unified 31D architecture)
        self.grasp_object = self.scene.rigid_objects["grasp_object"]

        # Number of bodies for force buffers
        self.num_bodies = self.robot.num_bodies

        # --- Inner-loop controller ---
        self.attitude_ctrl = AttitudeController(
            num_envs=self.num_envs,
            device=self.device,
        )

        # --- State buffers ---
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.raw_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_pos_err = torch.zeros(self.num_envs, device=self.device)  # for progress reward

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

        # NOTE: Observation normalization is handled by SKRL's RunningStandardScaler
        # in the PPO agent config. Do NOT normalize here to avoid double normalization.

        # --- Object/grasping state (always initialized for unified 31D obs) ---
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_mass = torch.zeros(self.num_envs, device=self.device)
        self.is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.object_init_z = torch.zeros(self.num_envs, device=self.device)  # ground rest height
        self.contain_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._box_dynamic = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.grasp_offset = torch.zeros(self.num_envs, 3, device=self.device)  # Stage 4: random grasp offset


        # --- Goal visualization marker ---
        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.08, 0.08, 0.08)
        marker_cfg.prim_path = "/Visuals/Command/goal_position"
        self.goal_marker = VisualizationMarkers(marker_cfg)

        # --- Action scaling ---
        self._setup_action_scaling()

    def _setup_action_scaling(self):
        """Define action space scaling from [-1, 1] to physical units.

        Always 8D: [ax, ay, az, wx, wy, wz, yaw_ref, gripper].
        """
        self.action_low = torch.tensor(
            [-8.0, -8.0, -8.0, -3.0, -3.0, -3.0, -math.pi, -0.087266],
            device=self.device,
        )
        self.action_high = torch.tensor(
            [8.0, 8.0, 8.0, 3.0, 3.0, 3.0, math.pi, 0.872665],
            device=self.device,
        )

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

        # Grasp object for Stage 3+: InteractiveScene auto-creates RigidObject
        # from the GripperDroneSceneCfg.grasp_object field. No manual creation needed.

        # Ground plane via TerrainImporter (following Crazyflie example pattern)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ========================================================================
    # Pre-physics: process RL actions through inner-loop controller
    # ========================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process RL actions. Called once per RL step (150 Hz)."""
        # Store previous action BEFORE overwriting with current (used in reward smoothness)
        self.prev_action = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        self.scaled_actions = self._scale_action(actions)

        # Gripper plate commands (always 8D action space)
        if self.cfg.lock_gripper:
            # Default: gripper fully open
            gripper_cmd = torch.full((self.num_envs,), 0.873, device=self.device)
            # Auto-close on sustained dock (contain_hold_count tracks cumulative containment)
            dock_threshold = 100 if self.cfg.stage == Stage.LOADED_FLIGHT else 150
            docked_mask = self.contain_hold_count >= dock_threshold
            gripper_cmd[docked_mask] = -0.087  # fully closed
        else:
            # Policy controls gripper
            gripper_cmd = self.scaled_actions[:, 7]
        plate_targets = torch.stack([gripper_cmd, gripper_cmd], dim=-1)
        self.robot.set_joint_position_target(plate_targets, joint_ids=self.plate_joint_ids)

        # Stage 4: controller mass already set in _execute_grasp (includes payload)
        # No freeze needed — _execute_grasp handles physical grasping inside _reset_idx

    def _apply_action(self):
        """Apply forces/torques to robot. Called every simulation step (300 Hz).

        IMPORTANT: Isaac Lab's set_external_force_and_torque() expects forces in each
        body's LOCAL frame (not world frame). Motor forces are already in body frame.
        World-frame disturbances (wind, gravity) must be rotated into body frame via R^T.
        """
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

        self.ext_forces.zero_()
        self.ext_torques.zero_()

        # Domain randomization: scale motor thrust by k_f scale per env
        kf_scale = self.motor_kf_scale.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        forces_b = forces_b * kf_scale
        torques_b = torques_b * kf_scale

        # Motor forces/torques are already in body-local frame — apply directly
        for i in range(4):
            motor_id = self.motor_body_ids[i]
            self.ext_forces[:, motor_id, :] = forces_b[:, i, :]
            self.ext_torques[:, motor_id, :] = torques_b[:, i, :]

        # --- World-frame disturbances → transform to body frame for base_link ---
        # Accumulate all world-frame forces, then rotate once with R^T
        world_force = torch.zeros(self.num_envs, 3, device=self.device)

        # Mass scale: extra gravity from mass variation
        nominal_mass = 1.080
        world_force[:, 2] -= (self.mass_scale - 1.0) * nominal_mass * 9.81

        # Wind disturbance (world frame)
        self._update_wind()
        world_force += self.wind_force

        # Payload weight: handled by PhysX (dynamic box in gripper, gravity ON)
        # No manual force/torque — real physics contact provides weight+torque

        # Rotate world → body frame: f_body = R^T @ f_world
        R = quat_to_rot_matrix(quat_w)  # (N, 3, 3)
        body_force = torch.bmm(R.transpose(1, 2), world_force.unsqueeze(-1)).squeeze(-1)
        self.ext_forces[:, self.base_body_id[0], :] += body_force

        # Apply to simulation
        self.robot.set_external_force_and_torque(self.ext_forces, self.ext_torques)

        # Physics-based grasping: NO teleport. PhysX handles contact between
        # gripper plates and box. The policy must physically close the plates
        # around the object to hold it.

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
        """Compute observation tensor — fully gripper-centric.

        All position/velocity observations reference the GRIPPER center (14.85cm below body),
        not the drone body. This ensures observation-reward consistency: the policy sees
        "where is the target relative to my gripper" and is rewarded for gripper precision.

        31D = [gripper_vel_b(3), ang_vel_b(3), R(9), p_goal_g(3), p_err_g_w(3),
               prev_action_xy(2), theta_plates(2), p_obj_g(3), grasp_flag(1),
               payload_est(1), d_obj_g(1)]
        """
        # Base state
        pos_w = self.robot.data.root_pos_w                # (N, 3)
        quat_w = self.robot.data.root_quat_w              # (N, 4) wxyz
        vel_b = self.robot.data.root_lin_vel_b             # (N, 3)
        ang_vel_b = self.robot.data.root_ang_vel_b         # (N, 3)
        R = quat_to_rot_matrix(quat_w)                     # (N, 3, 3)

        # --- Gripper position in world frame ---
        gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=self.device)
        gripper_offset_w = torch.bmm(
            R, gripper_offset_b.expand(self.num_envs, 3).unsqueeze(-1)
        ).squeeze(-1)
        gripper_pos_w = pos_w + gripper_offset_w

        # --- Gripper velocity in body frame ---
        # v_gripper = v_body + omega × r_offset (in body frame)
        gripper_vel_b = vel_b.clone()
        gripper_vel_b[:, 0] -= ang_vel_b[:, 1] * 0.08  # vx -= wy * r
        gripper_vel_b[:, 1] += ang_vel_b[:, 0] * 0.08  # vy += wx * r
        # vz unchanged (offset is purely along z in body frame)

        # --- Goal relative to gripper ---
        p_err_g_w = self.goal_pos - gripper_pos_w  # world frame: gripper→goal
        p_goal_g = torch.bmm(R.transpose(1, 2), p_err_g_w.unsqueeze(-1)).squeeze(-1)

        # Flatten rotation matrix to 9D
        R_flat = R.reshape(self.num_envs, 9)

        # Add sensor noise
        gripper_vel_b_noisy = gripper_vel_b + self.cfg.domain_rand.vel_noise_std * torch.randn_like(gripper_vel_b)
        p_err_g_w_noisy = p_err_g_w + self.cfg.domain_rand.pos_noise_std * torch.randn_like(p_err_g_w)
        p_goal_g_noisy = torch.bmm(
            R.transpose(1, 2),
            p_err_g_w_noisy.unsqueeze(-1),
        ).squeeze(-1)

        # --- Always 31D: unified gripper-centric architecture ---

        # Update object position from physics (all stages read from sim)
        self.object_pos[:] = self.grasp_object.data.root_pos_w

        # Plate joint angles, normalized to [-1, 1]
        joint_pos = self.robot.data.joint_pos[:, self.plate_joint_ids]  # (N, 2)
        theta_norm = (joint_pos - 0.393) / 0.48

        # Object position relative to gripper (gripper frame)
        p_obj_err_g = self.object_pos - gripper_pos_w
        p_obj_g = torch.bmm(R.transpose(1, 2), p_obj_err_g.unsqueeze(-1)).squeeze(-1)
        p_obj_g += self.cfg.domain_rand.obj_detection_noise_std * torch.randn_like(p_obj_g)

        # Grasp flag, payload estimate, distance from gripper to object
        grasp_flag = self.is_grasped.float().unsqueeze(-1)
        payload_est = (self.payload_mass + 0.02 * torch.randn(self.num_envs, device=self.device)).unsqueeze(-1)
        d_obj_g = torch.norm(p_obj_err_g, dim=-1, keepdim=True)

        obs = torch.cat([
            gripper_vel_b_noisy,          # (N, 3) gripper velocity in body frame
            ang_vel_b,                    # (N, 3) angular velocity (same for rigid body)
            R_flat,                       # (N, 9) rotation matrix (same for rigid body)
            p_goal_g_noisy,               # (N, 3) goal in gripper frame
            p_err_g_w_noisy,              # (N, 3) gripper→goal error in world frame
            self.prev_action[:, :2],      # (N, 2) previous action xy
            theta_norm,                   # (N, 2) plate angles
            p_obj_g,                      # (N, 3) object in gripper frame
            grasp_flag,                   # (N, 1) grasped flag
            payload_est,                  # (N, 1) payload estimate
            d_obj_g,                      # (N, 1) gripper→object distance
        ], dim=-1)  # Total: 31D

        self.step_count += 1

        # Update goal marker visualization
        self.goal_marker.visualize(self.goal_pos)

        return {"policy": obs}

    # ========================================================================
    # Rewards
    # ========================================================================

    def _get_rewards(self) -> torch.Tensor:
        """Compute stage-appropriate rewards. All gripper-centric."""
        pos_w = self.robot.data.root_pos_w
        vel_b = self.robot.data.root_lin_vel_b
        vel_w = self.robot.data.root_lin_vel_w
        ang_vel_b = self.robot.data.root_ang_vel_b
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        # Gripper position (used by ALL stages)
        gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=self.device)
        gripper_offset_w = torch.bmm(
            R, gripper_offset_b.expand(pos_w.shape[0], 3).unsqueeze(-1)
        ).squeeze(-1)
        gripper_pos_w = pos_w + gripper_offset_w

        # Gripper velocity in body frame
        gripper_vel_b = vel_b.clone()
        gripper_vel_b[:, 0] -= ang_vel_b[:, 1] * 0.08
        gripper_vel_b[:, 1] += ang_vel_b[:, 0] * 0.08

        if self.cfg.stage == Stage.BASIC_FLIGHT:
            rewards, info = compute_stage1_rewards(
                gripper_pos_w=gripper_pos_w, vel_b=gripper_vel_b, rot_matrix=R,
                goal_w=self.goal_pos,
                action=self.raw_actions[:, :7],
                prev_action=self.prev_action[:, :7],
            )

            # Note: goal regeneration is handled by eval script, not during training.
            # Single goal per episode lets the policy learn precise approach.

        elif self.cfg.stage in (Stage.PRECISION_APPROACH, Stage.GRASPING):
            # Capture Column Containment reward
            # Only reward when box is inside the vertical column defined by
            # the gripper's open plates (plates tip = widest point) down to ground.
            from rewards.reward_fn import exp_reward

            pos_err = torch.norm(gripper_pos_w - self.object_pos, dim=-1)
            xy_err = torch.norm(gripper_pos_w[:, :2] - self.object_pos[:, :2], dim=-1)
            z_err = torch.abs(gripper_pos_w[:, 2] - self.object_pos[:, 2])

            # Box position in gripper local frame
            box_offset_w = self.object_pos - gripper_pos_w
            box_local = torch.bmm(R.transpose(1, 2), box_offset_w.unsqueeze(-1)).squeeze(-1)

            # ---- Soft Capture Column: plates-tip opening projected downward ----
            # Plates open at 50deg: tip Y = ±0.104m, X = ±0.05m (strut spacing)
            column_half_x = 0.05    # strut X spacing
            column_half_y = 0.104   # plate tip Y when fully open
            box_half = 0.04         # 8cm cube half-size

            # Soft column score: 1.0 at center, 0.0 at edge, smooth gradient
            x_contain = (1.0 - torch.abs(box_local[:, 0]) / column_half_x).clamp(0.0, 1.0)
            y_contain = (1.0 - torch.abs(box_local[:, 1]) / column_half_y).clamp(0.0, 1.0)
            z_below = (box_local[:, 2] < 0.02).float()  # box must be below gripper
            column_score = x_contain * y_contain * z_below  # 0~1, smooth

            # ---- Overlap at plates-tip level (precise containment) ----
            gripper_half_x = 0.05   # strut X spacing
            gripper_half_y = 0.062  # plate Y at strut level (narrower than tips)

            box_min_x = box_local[:, 0] - box_half
            box_max_x = box_local[:, 0] + box_half
            box_min_y = box_local[:, 1] - box_half
            box_max_y = box_local[:, 1] + box_half

            overlap_x = (torch.min(torch.full_like(box_max_x, gripper_half_x), box_max_x)
                       - torch.max(torch.full_like(box_min_x, -gripper_half_x), box_min_x)).clamp(min=0.0)
            overlap_y = (torch.min(torch.full_like(box_max_y, gripper_half_y), box_max_y)
                       - torch.max(torch.full_like(box_min_y, -gripper_half_y), box_min_y)).clamp(min=0.0)

            box_area = (2 * box_half) ** 2
            overlap_xy = (overlap_x * overlap_y) / box_area

            # Z gating: box within strut vertical range
            z_in_range = ((box_local[:, 2] > -0.12) & (box_local[:, 2] < 0.02)).float()
            overlap_ratio = overlap_xy * z_in_range

            full_contain = (overlap_ratio > 0.50).float()

            # ========== Axis 1: Approach (split XY/Z, Z above box only) ==========
            above_box = (gripper_pos_w[:, 2] > self.object_pos[:, 2]).float()
            r_approach_xy = 3.0 * torch.exp(-1.5 * xy_err)
            r_approach_z = 2.0 * torch.exp(-2.0 * z_err) * above_box
            r_approach = r_approach_xy + r_approach_z

            # ========== Axis 2: Column alignment (soft gate, decomposed) ==========
            approach_gate = ((0.25 - xy_err) / 0.20).clamp(0.0, 1.0)
            r_x_align = 1.5 * x_contain * approach_gate
            r_y_align = 1.5 * y_contain * approach_gate
            r_z_below = 2.0 * z_below * approach_gate

            # Staged descent: XY first, then descend
            xy_aligned = (xy_err < 0.10).float()
            alt_target = self.object_pos[:, 2] + 0.25 * (1.0 - xy_aligned)
            staged_z_err = torch.abs(gripper_pos_w[:, 2] - alt_target)
            r_z = 2.0 * torch.exp(-2.0 * staged_z_err)

            # Fine approach: bridges 5-15cm gap
            r_fine = 5.0 * torch.exp(-10.0 * xy_err) * approach_gate * above_box

            # ========== Axis 3: Precision dock (overlap-based) ==========
            r_contain = 15.0 * overlap_ratio + 10.0 * full_contain

            is_contained = (overlap_ratio > 0.50)
            r_dock_bonus = 10.0 * is_contained.float()

            self.contain_hold_count += is_contained.long()
            hold_duration = (self.contain_hold_count.float() / 150.0).clamp(max=3.0)
            r_hold_stable = 10.0 * hold_duration * is_contained.float()

            contain_success = (self.contain_hold_count >= 225).float()
            r_success = 50.0 * contain_success

            # ========== Penalties ==========
            action7 = self.raw_actions[:, :7]
            prev7 = self.prev_action[:, :7]
            r_smooth = -0.2 * torch.sum((action7 - prev7) ** 2, dim=-1)
            r_mag = -0.1 * torch.sum(action7 ** 2, dim=-1)

            # Tilt
            tilt_angle = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
            r_tilt_descent = -3.0 * tilt_angle * xy_aligned
            near_dock = (overlap_ratio > 0.3).float()
            r_tilt_dock = -2.0 * tilt_angle * near_dock

            r_column_total = r_x_align + r_y_align + r_z_below
            rewards = (r_approach + r_column_total + r_z + r_fine
                       + r_contain + r_dock_bonus + r_hold_stable + r_success
                       + r_tilt_descent + r_tilt_dock
                       + r_smooth + r_mag)
            info = {
                "r_approach": r_approach, "r_approach_xy": r_approach_xy, "r_approach_z": r_approach_z,
                "r_column": r_column_total,
                "r_x_align": r_x_align, "r_y_align": r_y_align, "r_z_below": r_z_below,
                "r_z": r_z, "r_fine": r_fine,
                "r_contain": r_contain, "r_dock_bonus": r_dock_bonus,
                "r_hold_stable": r_hold_stable, "r_success": r_success,
                "r_tilt_descent": r_tilt_descent, "r_tilt_dock": r_tilt_dock,
                "r_smooth": r_smooth, "r_mag": r_mag,
                "xy_err": xy_err, "alt_err": z_err,
                "gripper_xy_offset": xy_err, "pos_error": pos_err,
                "overlap_ratio": overlap_ratio, "full_contain": full_contain,
                "column_score": column_score, "approach_gate": approach_gate,
            }

        elif self.cfg.stage == Stage.LOADED_FLIGHT:
            # Grasping already completed in _execute_grasp (inside _reset_idx).
            # RL only sees grasped state — same as Stage 1 but with payload.
            rewards, info = compute_stage1_rewards(
                gripper_pos_w=gripper_pos_w, vel_b=gripper_vel_b,
                rot_matrix=R, goal_w=self.goal_pos,
                action=self.raw_actions, prev_action=self.prev_action,
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

        # Log reward components and metrics to TensorBoard via SKRL
        # SKRL reads infos["episode"] dict — each value must be a scalar tensor
        log_dict = {}
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                log_dict[k] = torch.tensor(v.mean().item())
        # Add key flight metrics
        log_dict["mean_speed"] = torch.tensor(torch.norm(vel_w, dim=-1).mean().item())
        log_dict["mean_tilt_deg"] = torch.tensor(
            (torch.acos(R[:, 2, 2].clamp(-1, 1)) * 180 / 3.14159).mean().item()
        )
        self.extras["log"] = {k: v.mean().item() for k, v in info.items()}
        self.extras["episode"] = log_dict

        return rewards

    def _check_grasp_conditions(
        self,
        pos_w: torch.Tensor,
        vel_w: torch.Tensor,
        R: torch.Tensor,
    ):
        """Check and update grasp state for Stage 3 (physics-based).

        Physics-based grasp detection:
        - Box is lifted off ground (z > init_z + 0.04m) = something is holding it
        - Plates are closed (< 0.1 rad)
        - Sustained for 15+ steps (0.1s) = not a momentary bounce
        With gravity ON, box cannot float unless gripper holds it.

        Drop detection: box returns to ground level.
        """
        # Use ACTUAL physics position of the box
        actual_obj_pos = self.grasp_object.data.root_pos_w  # (N, 3)
        self.object_pos[:] = actual_obj_pos

        # Compute gripper position for reward/observation use
        gripper_offset_b = torch.tensor([0.0, 0.0, -0.08], device=self.device)
        gripper_offset_w = torch.bmm(
            R, gripper_offset_b.expand(pos_w.shape[0], 3).unsqueeze(-1)
        ).squeeze(-1)
        gripper_pos_w = pos_w + gripper_offset_w

        plate_angle = self.robot.data.joint_pos[:, self.plate_joint_ids[0]]
        plates_closed = plate_angle < self.cfg.grasp_plate_threshold  # < 0.1 rad

        # Sustained proximity grasp detection: gripper close + plates closed for 10+ steps
        # Prevents false triggers from momentary contact/bumps
        d_obj = torch.norm(gripper_pos_w - actual_obj_pos, dim=-1)
        grasp_condition = (d_obj < 0.10) & plates_closed
        self.lift_count = torch.where(
            grasp_condition,
            self.lift_count + 1,
            torch.zeros_like(self.lift_count),
        )
        trigger = (self.lift_count > 10) & (~self.is_grasped)
        self.is_grasped |= trigger

        # Drop detection: gripper moved away from object
        if self.is_grasped.any():
            drop = self.is_grasped & (d_obj > 0.20)
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
        # Stage 4: higher min altitude, only after grasping complete
        if self.cfg.stage == Stage.LOADED_FLIGHT:
            min_alt = 0.40
            grasped_s4 = self.contain_hold_count >= 100
            too_low = (pos_w[:, 2] < min_alt) & grasped_s4
        else:
            too_low = pos_w[:, 2] < self.cfg.min_altitude
        too_far = torch.norm(pos_w[:, :2], dim=-1) > self.cfg.max_distance
        too_tilted = tilt > max_tilt

        # Box fell off pedestal (Stage 3 with dynamic box)
        # Only terminate if gripper hasn't docked (if docked, box may be lifted)
        dock_threshold = 100 if self.cfg.stage == Stage.LOADED_FLIGHT else 150
        box_fell = (self.object_pos[:, 2] < 0.30) & (self.contain_hold_count < dock_threshold)

        # Stage 4: box must stay in gripper — terminate if lost
        # Only check after gripper has physically closed (contain_hold_count >= threshold)
        if self.cfg.stage == Stage.LOADED_FLIGHT:
            gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=self.device)
            gripper_pos = pos_w + torch.bmm(R, gripper_offset.expand(self.num_envs, 3).unsqueeze(-1)).squeeze(-1)
            box_dist = torch.norm(self.object_pos - gripper_pos, dim=-1)
            grasped = self.contain_hold_count >= 100
            box_lost = (box_dist > 0.15) & grasped  # only after real grasp
        else:
            box_lost = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        terminated = too_low | too_far | too_tilted | box_fell | box_lost

        # Truncation (time limit)
        max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        truncated = self.step_count >= max_steps
        # Stage 3: also truncate on successful 3s containment
        if self.cfg.stage == Stage.GRASPING:
            contain_success = self.contain_hold_count >= 225
            truncated = truncated | contain_success

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
        spawn_offset = self.cfg.spawn_spread * (torch.rand(num_reset, 3, device=self.device) - 0.5)
        spawn_offset[:, 2] *= 0.5  # Less Z variation

        # --- Reset grasping state and sample objects (all stages) ---
        self.is_grasped[env_ids] = False
        self.was_grasped[env_ids] = False
        self.lift_count[env_ids] = 0
        self.contain_hold_count[env_ids] = 0
        self._box_dynamic[env_ids] = False
        self._sample_objects(env_ids)
        self.object_init_z[env_ids] = self.object_pos[env_ids, 2]  # ground rest z

        # Reset physical object in simulation
        obj_state = torch.zeros(num_reset, 13, device=self.device)
        obj_state[:, :3] = self.object_pos[env_ids]
        obj_state[:, 3] = 1.0  # quaternion w
        self.grasp_object.write_root_state_to_sim(obj_state, env_ids)

        # --- Spawn drone ---
        if self.cfg.stage == Stage.LOADED_FLIGHT:
            # Stage 4: spawn drone directly above box (no offset)
            root_pos = self.object_pos[env_ids].clone()
            root_pos[:, 2] += 0.08  # gripper center offset → body above box
        elif self.cfg.stage == Stage.GRASPING and self.cfg.grasping_phase == 1:
            # Stage 3 Phase 1: spawn drone directly above box for precise dock learning
            root_pos = self.object_pos[env_ids].clone()
            root_pos[:, 2] = 0.85  # 30cm above box top (box z=0.54)
            # small XY perturbation: ±5cm
            root_pos[:, :2] += 0.10 * (torch.rand(num_reset, 2, device=self.device) - 0.5)
        elif self.cfg.stage.value >= Stage.PRECISION_APPROACH.value:
            default_pos = torch.tensor([0.0, 0.0, 1.5], device=self.device)
            root_pos = default_pos + spawn_offset
        else:
            default_pos = torch.tensor([0.0, 0.0, 3.0], device=self.device)
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
        # Set plate joints
        if self.cfg.stage == Stage.LOADED_FLIGHT:
            # Stage 4: plates closed — _setup_grasped_state also sets joint target
            for jid in self.plate_joint_ids:
                joint_pos[:, jid] = -0.087  # closed position
        else:
            # Default: plates at 45 deg (open/landing)
            for jid in self.plate_joint_ids:
                joint_pos[:, jid] = 0.785398
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # --- Reset controller ---
        self.attitude_ctrl.reset(env_ids)
        self.attitude_ctrl.mass[env_ids] = self.attitude_ctrl.base_mass  # reset to drone-only mass

        # --- Reset task state ---
        self.step_count[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.raw_actions[env_ids] = 0.0

        # Sample new goal
        self._sample_goals(env_ids)

        # Initialize prev_pos_err for progress reward (distance from spawn to goal)
        self.prev_pos_err[env_ids] = torch.norm(
            self.goal_pos[env_ids] - root_pos, dim=-1
        )

        # Stage 4: physically grasp box BEFORE RL loop sees this env.
        # Run sim steps inside reset to close gripper around box.
        # RL only ever sees the grasped state — no Phase 1 pollution.
        if self.cfg.stage == Stage.LOADED_FLIGHT:
            self.payload_mass[env_ids] = 0.2
            self._setup_grasped_state(env_ids)

    def _setup_grasped_state(self, env_ids: torch.Tensor):
        """Set up pre-grasped state for Stage 4. No sim stepping.

        Plates are set to closed position via write_joint_state AND joint target.
        contain_hold_count is set high so _pre_physics_step maintains closed target.
        On the very first sim step, the actuator exerts holding force at the closed position.
        """
        # Set contain_hold_count so gripper stays closed in _pre_physics_step
        self.contain_hold_count[env_ids] = 200
        self.is_grasped[env_ids] = True
        self.was_grasped[env_ids] = True

        # Explicitly set joint position target to closed BEFORE first sim step
        # This ensures actuator exerts holding force from step 1
        closed_target = torch.full((len(env_ids), 2), -0.087, device=self.device)
        self.robot.set_joint_position_target(closed_target, joint_ids=self.plate_joint_ids, env_ids=env_ids)

        # Update controller mass with payload
        drone_mass = self.attitude_ctrl.base_mass * self.mass_scale[env_ids]
        self.attitude_ctrl.mass[env_ids] = drone_mass + self.payload_mass[env_ids]

    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample goal positions based on current stage."""
        n = len(env_ids)

        if self.cfg.stage == Stage.BASIC_FLIGHT:
            xy = self.cfg.goal_pos_range_xy * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
            z_low, z_high = self.cfg.goal_pos_range_z
            z = torch.empty(n, 1, device=self.device).uniform_(z_low, z_high)
            self.goal_pos[env_ids] = torch.cat([xy, z], dim=-1)

        elif self.cfg.stage == Stage.PRECISION_APPROACH:
            # Goal = object position (gripper-centric: gripper targets object directly)
            self.goal_pos[env_ids] = self.object_pos[env_ids].clone()

        elif self.cfg.stage in (Stage.GRASPING, Stage.RELEASE):
            # Goal = object position (consistent with Stage 2)
            self.goal_pos[env_ids] = self.object_pos[env_ids].clone()

        elif self.cfg.stage == Stage.LOADED_FLIGHT:
            xy = self.cfg.delivery_range_xy * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)
            z = torch.full((n, 1), self.cfg.delivery_z, device=self.device)
            self.goal_pos[env_ids] = torch.cat([xy, z], dim=-1)

    def _sample_objects(self, env_ids: torch.Tensor):
        """Sample object positions and properties. Called for ALL stages."""
        n = len(env_ids)
        xy = self.cfg.goal_pos_range_xy * 0.5 * (torch.rand(n, 2, device=self.device) * 2.0 - 1.0)

        if self.cfg.stage == Stage.LOADED_FLIGHT:
            # Stage 4: box at gripper center (drone spawns at z=0.62, gripper at z=0.54)
            z = torch.full((n, 1), 0.54, device=self.device)
            xy[:] = 0.0  # box directly under drone (same XY as drone spawn)
        elif self.cfg.stage in (Stage.BASIC_FLIGHT, Stage.PRECISION_APPROACH, Stage.GRASPING):
            # All approach/grasp stages: box on pedestal
            z = torch.full((n, 1), 0.54, device=self.device)
        else:
            # Stage 5: object at target altitude range
            z_low, z_high = self.cfg.target_z
            z = torch.empty(n, 1, device=self.device).uniform_(z_low, z_high)

        self.object_pos[env_ids] = torch.cat([xy, z], dim=-1)

        # Move pedestal to match box XY position (if pedestal exists)
        if hasattr(self.scene, 'rigid_objects') and 'pedestal' in self.scene.rigid_objects:
            pedestal = self.scene.rigid_objects["pedestal"]
            ped_state = torch.zeros(n, 13, device=self.device)
            ped_state[:, 0] = xy[:, 0]  # match box X
            ped_state[:, 1] = xy[:, 1]  # match box Y
            ped_state[:, 2] = 0.25      # pedestal center z
            ped_state[:, 3] = 1.0       # quaternion w (identity)
            pedestal.write_root_state_to_sim(ped_state, env_ids)

        # For Stage 3+, goal_pos is set to object position (overwritten by _sample_goals for Stage 1-2)
        if self.cfg.stage in (Stage.GRASPING, Stage.RELEASE):
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
