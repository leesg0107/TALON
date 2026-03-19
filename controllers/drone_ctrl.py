"""
Attitude controller and motor model for the Gripper-Drone quadrotor.

All operations are batched over N environments using PyTorch tensors.
Runs at the inner-loop rate (300 Hz) while the RL policy runs at 150 Hz.

Control flow:
  RL action (7D) -> action scaling -> desired accel/rates
  -> attitude controller -> desired thrusts per motor
  -> motor dynamics (first-order lag) -> actual RPM
  -> thrust model -> forces/torques in world frame
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class MotorParams:
    """Physical parameters for the quadrotor motor/propeller system."""
    # Thrust coefficient: F = k_f * omega^2  [N / (rad/s)^2]
    k_f: float = 1.0e-5
    # Torque coefficient: tau = k_m * omega^2  [N*m / (rad/s)^2]
    k_m: float = 1.6e-7
    # Motor time constant [s] (first-order lag)
    tau_motor: float = 0.02
    # Motor speed limits [rad/s]
    omega_min: float = 100.0
    omega_max: float = 838.0
    # Motor arm length from CoM [m] (computed from URDF: sqrt(0.123744^2 + 0.123744^2))
    arm_length: float = 0.175
    # Motor positions in body frame (from URDF, X-config at 45 deg)
    # Motor 0: front-right (+x, +y), Motor 1: front-left (-x, +y)
    # Motor 2: back-left  (-x, -y), Motor 3: back-right (+x, -y)
    # Spin directions: 0=CW(-), 1=CCW(+), 2=CW(-), 3=CCW(+)


class MotorModel:
    """First-order motor dynamics and thrust computation.

    Models motor response lag and converts RPM to forces/torques.
    All operations batched over (num_envs, 4) motor dimensions.
    """

    def __init__(self, num_envs: int, device: torch.device, params: MotorParams | None = None):
        self.num_envs = num_envs
        self.device = device
        self.params = params or MotorParams()

        # Current motor speeds [rad/s], initialized at hover
        hover_omega = self._hover_omega()
        self.omega = torch.full((num_envs, 4), hover_omega, device=device)

        # Motor positions in body frame [m] (from URDF)
        d = 0.123744
        self.motor_pos_b = torch.tensor([
            [+d, +d, 0.0365],  # motor_0: front-right
            [-d, +d, 0.0365],  # motor_1: front-left
            [-d, -d, 0.0365],  # motor_2: back-left
            [+d, -d, 0.0365],  # motor_3: back-right
        ], device=device)  # (4, 3)

        # Spin direction signs: CW = -1, CCW = +1 (for yaw torque)
        self.spin_dir = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device)

        # Precompute mixer matrix: [T, tau_x, tau_y, tau_z] = M @ [F0, F1, F2, F3]
        # where F_i = k_f * omega_i^2
        y = self.motor_pos_b[:, 1]  # y-coordinates
        x = self.motor_pos_b[:, 0]  # x-coordinates
        self.mixer = torch.tensor([
            [1.0, 1.0, 1.0, 1.0],                # Total thrust
            [y[0], y[1], y[2], y[3]],              # Roll torque (from y-offset * thrust)
            [-x[0], -x[1], -x[2], -x[3]],         # Pitch torque (from -x-offset * thrust)
            [self.params.k_m / self.params.k_f * s for s in self.spin_dir],  # Yaw torque
        ], device=device)  # (4, 4)

        # Inverse mixer for allocation: [F0, F1, F2, F3] = M_inv @ [T, tau_x, tau_y, tau_z]
        self.mixer_inv = torch.linalg.inv(self.mixer)

    def _hover_omega(self) -> float:
        """Compute hover motor speed assuming 1.08 kg total mass."""
        import math
        hover_thrust_per_motor = 1.08 * 9.81 / 4.0
        return math.sqrt(hover_thrust_per_motor / self.params.k_f)

    def reset(self, env_ids: torch.Tensor):
        """Reset motor speeds to hover for specified environments."""
        hover_omega = self._hover_omega()
        self.omega[env_ids] = hover_omega

    def step(self, omega_cmd: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance motor dynamics by one timestep.

        Args:
            omega_cmd: Commanded motor speeds (num_envs, 4) [rad/s]
            dt: Timestep [s]

        Returns:
            forces_b: Per-motor thrust forces in body frame (num_envs, 4, 3)
            torques_b: Reaction torques per motor in body frame (num_envs, 4, 3)
        """
        # Clamp commands
        omega_cmd = omega_cmd.clamp(self.params.omega_min, self.params.omega_max)

        # First-order lag: omega += (omega_cmd - omega) * dt / tau
        alpha = dt / self.params.tau_motor
        self.omega = self.omega + alpha * (omega_cmd - self.omega)
        self.omega.clamp_(self.params.omega_min, self.params.omega_max)

        # Thrust per motor: F = k_f * omega^2, along body +Z
        omega_sq = self.omega ** 2
        thrust = self.params.k_f * omega_sq  # (N, 4)

        # Forces in body frame: each motor pushes along +Z
        forces_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        forces_b[:, :, 2] = thrust

        # Reaction torques: tau = k_m * omega^2 * spin_direction, about body Z
        torques_b = torch.zeros(self.num_envs, 4, 3, device=self.device)
        torques_b[:, :, 2] = self.params.k_m * omega_sq * self.spin_dir.unsqueeze(0)

        return forces_b, torques_b

    def allocate(self, wrench_cmd: torch.Tensor) -> torch.Tensor:
        """Convert desired [T, tau_x, tau_y, tau_z] to motor speeds.

        Args:
            wrench_cmd: (num_envs, 4) desired [thrust, roll_torque, pitch_torque, yaw_torque]

        Returns:
            omega_cmd: (num_envs, 4) motor speed commands [rad/s]
        """
        # F_i = mixer_inv @ wrench
        thrust_per_motor = (self.mixer_inv.unsqueeze(0) @ wrench_cmd.unsqueeze(-1)).squeeze(-1)
        # F = k_f * omega^2  =>  omega = sqrt(F / k_f)
        thrust_per_motor = thrust_per_motor.clamp(min=0.0)
        omega_cmd = torch.sqrt(thrust_per_motor / self.params.k_f)
        return omega_cmd.clamp(self.params.omega_min, self.params.omega_max)


class AttitudeController:
    """Simplified INDI-like attitude controller for quadrotor.

    Converts RL policy outputs (desired acceleration + body rates + yaw ref)
    into motor speed commands. Runs at inner-loop rate (300 Hz).

    The controller:
    1. Converts desired body-frame acceleration to desired attitude + total thrust
    2. Computes attitude error and applies PD control for body torques
    3. Uses the mixer to convert [thrust, torques] to motor commands
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        total_mass: float = 1.08,
        gravity: float = 9.81,
        kp_att: tuple[float, float, float] = (8.0, 8.0, 4.0),
        kd_att: tuple[float, float, float] = (1.5, 1.5, 0.8),
    ):
        self.num_envs = num_envs
        self.device = device
        self.mass = total_mass
        self.gravity = gravity

        self.kp = torch.tensor(kp_att, device=device)
        self.kd = torch.tensor(kd_att, device=device)

        self.motor_model = MotorModel(num_envs, device)

    def reset(self, env_ids: torch.Tensor):
        """Reset controller state for specified environments."""
        self.motor_model.reset(env_ids)

    def compute(
        self,
        accel_cmd_b: torch.Tensor,     # (N, 3) desired linear acceleration in body frame
        rate_cmd_b: torch.Tensor,       # (N, 3) desired body angular rates [rad/s]
        yaw_ref: torch.Tensor,          # (N,) desired yaw angle [rad]
        quat_w: torch.Tensor,           # (N, 4) current quaternion [w, x, y, z]
        ang_vel_b: torch.Tensor,        # (N, 3) current body angular velocity
        dt: float,                       # timestep [s]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute motor forces from high-level RL commands.

        Returns:
            forces_b: per-motor forces in body frame (N, 4, 3)
            torques_b: per-motor reaction torques in body frame (N, 4, 3)
        """
        N = self.num_envs

        # --- Step 1: Compute desired thrust magnitude ---
        # In body frame, the thrust along Z must compensate gravity + provide desired accel
        # Gravity in body frame: g_b = R_wb^T @ [0, 0, -g]
        R = quat_to_rot_matrix(quat_w)  # (N, 3, 3)
        g_world = torch.tensor([0.0, 0.0, -self.gravity], device=self.device)
        g_body = torch.bmm(R.transpose(1, 2), g_world.expand(N, 3).unsqueeze(-1)).squeeze(-1)

        # Required specific force in body frame (acceleration - gravity)
        # F_body / m = accel_cmd_b - g_body  (g_body is negative-Z when level)
        specific_force_b = accel_cmd_b - g_body  # (N, 3)

        # Total thrust = mass * body-Z component of specific force
        # We also add a small XY-to-attitude coupling for better tracking
        total_thrust = self.mass * specific_force_b[:, 2].clamp(min=0.5, max=4.0 * self.gravity)

        # --- Step 2: Compute desired attitude from desired acceleration ---
        # Desired body Z-axis in world frame (thrust direction)
        accel_world = torch.bmm(R, accel_cmd_b.unsqueeze(-1)).squeeze(-1)
        des_accel_world = accel_world + torch.tensor([0.0, 0.0, self.gravity], device=self.device)
        des_z_world = des_accel_world / (des_accel_world.norm(dim=-1, keepdim=True) + 1e-6)

        # Desired yaw defines the heading
        cos_yaw = torch.cos(yaw_ref)
        sin_yaw = torch.sin(yaw_ref)
        des_x_c = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(yaw_ref)], dim=-1)

        # Construct desired rotation matrix
        des_y = torch.cross(des_z_world, des_x_c, dim=-1)
        des_y = des_y / (des_y.norm(dim=-1, keepdim=True) + 1e-6)
        des_x = torch.cross(des_y, des_z_world, dim=-1)
        R_des = torch.stack([des_x, des_y, des_z_world], dim=-1)  # (N, 3, 3)

        # --- Step 3: Attitude error (SO(3) error) ---
        R_err = torch.bmm(R_des.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_des)
        att_err = 0.5 * torch.stack([
            R_err[:, 2, 1],  # e_x
            R_err[:, 0, 2],  # e_y
            R_err[:, 1, 0],  # e_z
        ], dim=-1)  # (N, 3) - vee map of skew-symmetric error

        # --- Step 4: PD body torque ---
        rate_err = rate_cmd_b - ang_vel_b
        body_torque = self.kp * att_err + self.kd * rate_err  # (N, 3)

        # --- Step 5: Allocate to motors ---
        wrench = torch.cat([
            total_thrust.unsqueeze(-1),
            body_torque,
        ], dim=-1)  # (N, 4)

        omega_cmd = self.motor_model.allocate(wrench)

        # --- Step 6: Motor dynamics ---
        forces_b, torques_b = self.motor_model.step(omega_cmd, dt)

        return forces_b, torques_b


def quat_to_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] to rotation matrix.

    Args:
        quat: (N, 4) quaternions in [w, x, y, z] format

    Returns:
        R: (N, 3, 3) rotation matrices
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)

    return R
