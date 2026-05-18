"""
GripperDroneEnv wrapper for waypoint training.

Trains Stage 1 (approach) and Stage 4 (loaded delivery) directly in GripperDroneEnv.
Eliminates WaypointDroneEnv → GripperDroneEnv transfer gap.

Stage 1: bypass_analytical=True, 22D obs, 4D action, waypoint rewards
Stage 4: Box pre-attached in gripper, pure loaded flight, 23D obs, 4D action
"""
from __future__ import annotations

import math
import torch
from isaaclab.envs import DirectRLEnv

from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.drone_env import GripperDroneEnv, quat_to_rot_matrix


class GripperWaypointEnv(DirectRLEnv):
    """Wrapper that makes GripperDroneEnv act like WaypointDroneEnv.

    Uses GripperDroneEnv for physics (same scene: box, pedestal, gripper, PD).
    Provides 22D/23D obs and 4D action like WaypointDroneEnv.
    """

    def __init__(self, cfg: GripperDroneEnvCfg, mode: str = "flight", **kwargs):
        self.mode = mode
        self._inner_cfg = cfg

        # Override obs/action spaces for 4D/22D
        if mode == "loaded":
            cfg.observation_space = 23
        else:
            cfg.observation_space = 22
        cfg.action_space = 4

        super().__init__(cfg, **kwargs)

        # Inner env references (reuse scene from DirectRLEnv)
        self.robot = self.scene["robot"]
        self.grasp_object = self.scene["grasp_object"]

        # Motor/body IDs
        motor_names = ["motor_0", "motor_1", "motor_2", "motor_3"]
        self.motor_body_ids = [self.robot.find_bodies(n)[0][0] for n in motor_names]
        self.base_body_id = self.robot.find_bodies("base_link")[0]
        self.plate_joint_ids = self.robot.find_joints("plate_joint.*")[0]

        # Attitude controller (same as GripperDroneEnv)
        from controllers.drone_ctrl import AttitudeController
        self.attitude_ctrl = AttitudeController(
            num_envs=self.num_envs, device=self.device,
        )

        # External forces
        num_bodies = self.robot.num_bodies
        self.ext_forces = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)
        self.ext_torques = torch.zeros(self.num_envs, num_bodies, 3, device=self.device)

        # Action/state buffers
        self.raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._steps_since_goal = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Trajectory: 4 sequential WPs per env
        self.NUM_WPS = 4
        self.trajectory = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._wp_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # DR
        self.mass_scale = torch.ones(self.num_envs, device=self.device)
        self.motor_kf_scale = torch.ones(self.num_envs, device=self.device)
        self.payload_mass = torch.zeros(self.num_envs, device=self.device)
        self.wind_force = torch.zeros(self.num_envs, 3, device=self.device)
        self.wind_phase = torch.zeros(self.num_envs, 3, device=self.device)

        # Object position cache (for loaded mode box tracking)
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._box_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Action scaling: 4D
        self.action_low = torch.tensor([-8.0, -8.0, -8.0, -math.pi], device=self.device)
        self.action_high = torch.tensor([8.0, 8.0, 8.0, math.pi], device=self.device)

    def _scale_action(self, raw_action):
        return self.action_low + (raw_action + 1.0) / 2.0 * (self.action_high - self.action_low)

    def _setup_scene(self):
        # Terrain (ground plane) — REQUIRED for env_spacing to work
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    # ========================================================================
    # Pre-physics
    # ========================================================================
    def _pre_physics_step(self, actions):
        self.prev_action = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        scaled = self._scale_action(actions)

        self._accel_cmd = scaled[:, :3]
        self._yaw_ref = scaled[:, 3]

        # Gripper: closed for loaded (box attached), open for flight
        if self.mode == "loaded":
            gripper_cmd = torch.full((self.num_envs,), -0.087, device=self.device)
        else:
            gripper_cmd = torch.full((self.num_envs,), 0.873, device=self.device)
        plate_targets = torch.stack([gripper_cmd, gripper_cmd], dim=-1)
        self.robot.set_joint_position_target(plate_targets, joint_ids=self.plate_joint_ids)

    def _apply_action(self):
        quat_w = self.robot.data.root_quat_w
        ang_vel_b = self.robot.data.root_ang_vel_b

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

        kf_scale = self.motor_kf_scale.unsqueeze(-1).unsqueeze(-1)
        forces_b = forces_b * kf_scale
        torques_b = torques_b * kf_scale

        for i in range(4):
            motor_id = self.motor_body_ids[i]
            self.ext_forces[:, motor_id, :] = forces_b[:, i, :]
            self.ext_torques[:, motor_id, :] = torques_b[:, i, :]

        world_force = torch.zeros(self.num_envs, 3, device=self.device)
        nominal_mass = 1.080
        world_force[:, 2] -= (self.mass_scale - 1.0) * nominal_mass * 9.81

        # Wind
        t = self.step_count.float() * self.cfg.sim.dt
        for axis in range(3):
            self.wind_force[:, axis] = 0.5 * torch.sin(
                2 * math.pi * 0.5 * t + self.wind_phase[:, axis])
        world_force += self.wind_force

        R = quat_to_rot_matrix(quat_w)
        body_force = torch.bmm(R.transpose(1, 2), world_force.unsqueeze(-1)).squeeze(-1)
        self.ext_forces[:, self.base_body_id[0], :] += body_force

        self.robot.set_external_force_and_torque(self.ext_forces, self.ext_torques)

    # ========================================================================
    # Observations
    # ========================================================================
    def _get_observations(self):
        pos_w = self.robot.data.root_pos_w
        vel_b = self.robot.data.root_lin_vel_b
        ang_vel_b = self.robot.data.root_ang_vel_b
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        goal_err_w = self.goal_pos - pos_w
        goal_b = torch.bmm(R.transpose(1, 2), goal_err_w.unsqueeze(-1)).squeeze(-1)
        R_flat = R.reshape(self.num_envs, 9)

        prev_norm = self.prev_action.clone()
        prev_norm[:, :3] /= 8.0
        prev_norm[:, 3] /= math.pi

        vel_b += 0.03 * torch.randn_like(vel_b)
        goal_b += 0.01 * torch.randn_like(goal_b)

        obs_parts = [vel_b, ang_vel_b, R_flat, goal_b, prev_norm]

        if self.mode == "loaded":
            pe = self.payload_mass.unsqueeze(-1) + 0.02 * torch.randn(self.num_envs, 1, device=self.device)
            obs_parts.append(pe)

        obs = torch.cat(obs_parts, dim=-1)
        self.step_count += 1
        return {"policy": obs}

    # ========================================================================
    # Rewards (same as WaypointDroneEnv v13)
    # ========================================================================
    def _get_rewards(self):
        pos_w = self.robot.data.root_pos_w
        vel_w = self.robot.data.root_lin_vel_w
        ang_vel_b = self.robot.data.root_ang_vel_b
        quat_w = self.robot.data.root_quat_w
        R = quat_to_rot_matrix(quat_w)

        rel_pos = self.goal_pos - pos_w
        dist_sq = torch.sum(rel_pos ** 2, dim=-1)
        pos_err = torch.sqrt(dist_sq).clamp(min=0.1)

        # Direction: velocity toward goal
        goal_dir = rel_pos / pos_err.unsqueeze(-1)
        vel_toward_goal = torch.sum(vel_w * goal_dir, dim=-1)
        r_direction = 0.5 * vel_toward_goal.clamp(-3.0, 3.0)

        # Time-based arrival
        self._steps_since_goal += 1
        arrived = (pos_err < 0.3).float()
        time_sec = self._steps_since_goal.float() / 150.0
        r_arrive = arrived * 10.0 / (time_sec + 0.5)

        # Crash (env-local Z)
        local_z = pos_w[:, 2] - self.scene.env_origins[:, 2]
        crashed = (local_z < 0.15).float()
        r_crash = -5.0 * crashed

        # Penalties
        action = self.raw_actions
        prev = self.prev_action
        r_smooth = -0.01 * torch.sum((action - prev) ** 2, dim=-1)
        ang_vel_norm = torch.norm(ang_vel_b, dim=-1)
        r_angular = -0.02 * ang_vel_norm

        # Tilt
        tilt_angle = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
        if self.mode == "loaded":
            tilt_excess = (tilt_angle - 0.45).clamp(min=0)  # ~26° threshold
            r_tilt = -3.0 * tilt_excess
        else:
            tilt_excess = (tilt_angle - 0.52).clamp(min=0)
            r_tilt = -3.0 * tilt_excess

        # Overshoot prevention: penalize high speed near WP (loaded mode only)
        # Speed floor at 1.5 m/s → model can approach at 1.5 without penalty
        if self.mode == "loaded":
            speed = torch.norm(vel_w, dim=-1)
            near_wp = (pos_err < 1.0).float()
            r_overshoot = -1.0 * near_wp * (speed - 1.5).clamp(min=0)

            # Tilt-gated anisotropic action barrier — only active when drone is tilted
            # high. At level flight (tilt < 26°), no penalty → policy navigates freely.
            # When tilt > 26°, soft quadratic penalty on excessive lateral cmds.
            # Anisotropic: X (strut, 1cm clearance) tighter than Y (plate, 6.4cm).
            # Weights tuned to NEVER exceed r_direction max (~1.5/step).
            a_x = action[:, 0]
            a_y = action[:, 1]
            tilt_gate = (tilt_angle > 0.45).float()        # ~26° gate
            r_barrier_x = -0.1 * tilt_gate * (a_x.abs() - 1.5).clamp(min=0) ** 2
            r_barrier_y = -0.05 * tilt_gate * (a_y.abs() - 2.5).clamp(min=0) ** 2
            r_action_barrier = r_barrier_x + r_barrier_y

            # Box tracking (loaded mode): detect if box fell
            self.object_pos[:] = self.grasp_object.data.root_pos_w
            box_local_z = self.object_pos[:, 2] - self.scene.env_origins[:, 2]
            self._box_dropped = box_local_z < 0.30
        else:
            r_overshoot = torch.zeros(self.num_envs, device=self.device)
            r_action_barrier = torch.zeros(self.num_envs, device=self.device)

        rewards = r_direction + r_arrive + r_crash + r_smooth + r_angular + r_tilt + r_overshoot + r_action_barrier

        # Timeout: 3s per WP → terminate (not just penalty)
        timed_out = (self._steps_since_goal > 450)
        self._wp_timeout = timed_out
        rewards = rewards - 2.0 * timed_out.float()

        # Box tracking (loaded mode): detect if box fell out of gripper
        if self.mode == "loaded":
            self.object_pos[:] = self.grasp_object.data.root_pos_w
            box_local_z = self.object_pos[:, 2] - self.scene.env_origins[:, 2]
            self._box_dropped = box_local_z < 0.30

        # WP reached → advance to next WP in trajectory
        reached = pos_err < 0.3
        if reached.any():
            reached_ids = reached.nonzero(as_tuple=False).view(-1)
            for eid in reached_ids.tolist():
                wi = self.wp_idx[eid].item() + 1
                if wi < self.NUM_WPS:
                    self.wp_idx[eid] = wi
                    self.goal_pos[eid] = self.trajectory[eid, wi]
                    self._steps_since_goal[eid] = 0
                else:
                    # Trajectory complete → generate new trajectory (continue episode)
                    self._generate_trajectory(torch.tensor([eid], device=self.device))

        return rewards

    def _generate_trajectory(self, env_ids):
        """Generate a trajectory of 4 sequential WPs in 3D.

        Each WP has independent random direction from the previous WP,
        creating varied paths (straight, zigzag, ascending, descending).
        """
        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        drone_pos = self.robot.data.root_pos_w[env_ids].clone()

        for i, eid in enumerate(env_ids.tolist()):
            origin = env_origins[i]
            prev = drone_pos[i].clone()

            if self.mode == "loaded":
                z_min, z_max = 0.8, 3.0
            else:
                z_min, z_max = 1.0, 3.5

            for j in range(self.NUM_WPS):
                # Each WP: random direction + distance from PREVIOUS WP
                angle = torch.rand(1).item() * 2 * math.pi
                dist = 0.8 + torch.rand(1).item() * 1.5  # 0.8~2.3m per segment
                wp_x = prev[0].item() + dist * math.cos(angle)
                wp_y = prev[1].item() + dist * math.sin(angle)
                wp_z = origin[2].item() + z_min + torch.rand(1).item() * (z_max - z_min)

                # Clamp XY to stay within env bounds
                local_x = wp_x - origin[0].item()
                local_y = wp_y - origin[1].item()
                local_x = max(-4.0, min(4.0, local_x))
                local_y = max(-4.0, min(4.0, local_y))
                wp_x = origin[0].item() + local_x
                wp_y = origin[1].item() + local_y

                self.trajectory[eid, j] = torch.tensor([wp_x, wp_y, wp_z], device=self.device)
                prev = self.trajectory[eid, j]

            self.wp_idx[eid] = 0
            self.goal_pos[eid] = self.trajectory[eid, 0]
            self._steps_since_goal[eid] = 0

    # ========================================================================
    # Termination
    # ========================================================================
    def _get_dones(self):
        pos_w = self.robot.data.root_pos_w
        R = quat_to_rot_matrix(self.robot.data.root_quat_w)

        # Use env-local position for termination checks
        local_pos = pos_w - self.scene.env_origins

        tilt = 1.0 - R[:, 2, 2]
        max_tilt = 1.0 - math.cos(math.radians(60))

        too_low = local_pos[:, 2] < 0.15
        too_far = torch.norm(local_pos[:, :2], dim=-1) > 10.0
        too_tilted = tilt > max_tilt

        # WP timeout → terminate (3s per WP, if not reached → episode over)
        wp_timeout = self._wp_timeout if hasattr(self, '_wp_timeout') else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Box dropped (loaded mode)
        if self.mode == "loaded":
            box_dropped = self._box_dropped if hasattr(self, '_box_dropped') else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            terminated = too_low | too_far | too_tilted | box_dropped | wp_timeout
        else:
            terminated = too_low | too_far | too_tilted | wp_timeout

        max_steps = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        truncated = self.step_count >= max_steps

        return terminated, truncated

    # ========================================================================
    # Reset
    # ========================================================================
    def _reset_idx(self, env_ids):
        num_reset = len(env_ids)

        # DR
        self.mass_scale[env_ids] = torch.empty(num_reset, device=self.device).uniform_(0.9, 1.1)
        self.motor_kf_scale[env_ids] = torch.empty(num_reset, device=self.device).uniform_(0.85, 1.15)
        self.wind_phase[env_ids] = torch.rand(num_reset, 3, device=self.device) * 2 * math.pi

        # Reset actions/state
        self.raw_actions[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self._steps_since_goal[env_ids] = 0
        self._box_dropped[env_ids] = False
        self._wp_timeout[env_ids] = False
        self.wp_idx[env_ids] = 0
        self.attitude_ctrl.reset(env_ids)

        # Get env origin offsets for world-frame positioning
        env_origins = self.scene.env_origins[env_ids]

        if self.mode == "loaded":
            # ============================================================
            # Stage 4: Spawn with box in gripper
            # Uses pre-simulated grasp states if available, otherwise idealized
            # ============================================================
            import os
            if not hasattr(self, '_grasp_states'):
                grasp_path = "data/grasp_states.pt"
                if os.path.exists(grasp_path):
                    self._grasp_states = torch.load(grasp_path, map_location=self.device)
                    print(f"[Stage4] Loaded {self._grasp_states['n_states'].item()} grasp states")
                else:
                    self._grasp_states = None

            self.payload_mass[env_ids] = torch.empty(num_reset, device=self.device).uniform_(0.15, 0.25)

            drone_mass = self.attitude_ctrl.base_mass * self.mass_scale[env_ids]
            self.attitude_ctrl.mass[env_ids] = drone_mass + self.payload_mass[env_ids]

            if self._grasp_states is not None:
                # === Use pre-simulated physically-realistic grasp states ===
                gs = self._grasp_states
                n_available = gs['n_states'].item()
                idx = torch.randint(0, n_available, (num_reset,), device=self.device)

                root_pos = env_origins + gs['drone_pos_local'][idx].to(self.device)
                root_quat = gs['drone_quat'][idx].to(self.device)
                root_vel = gs['drone_vel'][idx].to(self.device)

                root_state = torch.zeros(num_reset, 13, device=self.device)
                root_state[:, :3] = root_pos
                root_state[:, 3:7] = root_quat
                root_state[:, 7:] = root_vel
                self.robot.write_root_state_to_sim(root_state, env_ids)

                joint_pos = gs['joint_pos'][idx].to(self.device)
                joint_vel = torch.zeros_like(joint_pos)
                self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

                box_pos = env_origins + gs['box_pos_local'][idx].to(self.device)
                box_quat = gs['box_quat'][idx].to(self.device)
                obj_state = torch.zeros(num_reset, 13, device=self.device)
                obj_state[:, :3] = box_pos
                obj_state[:, 3:7] = box_quat
                self.grasp_object.write_root_state_to_sim(obj_state, env_ids)
            else:
                # === Fallback: idealized placement with offset ===
                root_pos = env_origins.clone()
                root_pos[:, 2] += 0.85
                root_pos[:, :2] += 0.10 * (torch.rand(num_reset, 2, device=self.device) - 0.5)

                small_euler = 0.05 * (torch.rand(num_reset, 3, device=self.device) - 0.5)
                root_quat = _euler_to_quat(small_euler)

                root_state = torch.zeros(num_reset, 13, device=self.device)
                root_state[:, :3] = root_pos
                root_state[:, 3:7] = root_quat
                self.robot.write_root_state_to_sim(root_state, env_ids)

                joint_pos = self.robot.data.joint_pos[env_ids].clone()
                joint_vel = torch.zeros_like(joint_pos)
                for jid in self.plate_joint_ids:
                    joint_pos[:, jid] = -0.087
                self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

                R = quat_to_rot_matrix(root_quat)
                gripper_offset = torch.tensor([0.0, 0.0, -0.08], device=self.device)
                gripper_offset = gripper_offset.expand(num_reset, 3).clone()
                gripper_offset[:, :2] += 0.02 * (2 * torch.rand(num_reset, 2, device=self.device) - 1)
                box_pos = root_pos + torch.bmm(R, gripper_offset.unsqueeze(-1)).squeeze(-1)
                box_euler = small_euler + 0.052 * (2 * torch.rand(num_reset, 3, device=self.device) - 1)
                box_quat = _euler_to_quat(box_euler)
                obj_state = torch.zeros(num_reset, 13, device=self.device)
                obj_state[:, :3] = box_pos
                obj_state[:, 3:7] = box_quat
                self.grasp_object.write_root_state_to_sim(obj_state, env_ids)

            # Generate trajectory of 4 sequential WPs
            self._generate_trajectory(env_ids)
        else:
            # ============================================================
            # Stage 1: normal flight spawn
            # ============================================================
            self.attitude_ctrl.mass[env_ids] = self.attitude_ctrl.base_mass * self.mass_scale[env_ids]

            spawn_offset = (torch.rand(num_reset, 3, device=self.device) - 0.5)
            local_pos = torch.tensor([0.0, 0.0, 3.0], device=self.device) + spawn_offset
            local_pos[:, 2] = local_pos[:, 2].clamp(min=1.5)
            root_pos = env_origins + local_pos

            small_euler = 0.05 * (torch.rand(num_reset, 3, device=self.device) - 0.5)
            root_quat = _euler_to_quat(small_euler)

            root_state = torch.zeros(num_reset, 13, device=self.device)
            root_state[:, :3] = root_pos
            root_state[:, 3:7] = root_quat
            self.robot.write_root_state_to_sim(root_state, env_ids)

            # Generate trajectory of 4 sequential WPs
            self._generate_trajectory(env_ids)


def _euler_to_quat(euler):
    r, p, y = euler[:, 0], euler[:, 1], euler[:, 2]
    cr, sr = torch.cos(r / 2), torch.sin(r / 2)
    cp, sp = torch.cos(p / 2), torch.sin(p / 2)
    cy, sy = torch.cos(y / 2), torch.sin(y / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_q = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y_q, z], dim=-1)
