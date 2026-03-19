"""
Reward functions for all 5 training stages of the Gripper-Drone.

All functions are pure torch, batched over (num_envs,).
Following the design spec: rewards use negative exponential kernel r = w * exp(-alpha * error).

Convention:
    All reward functions return (num_envs,) tensors of per-environment rewards.
    All inputs are batched along dim 0.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field


# ============================================================================
# Reward weight configurations per stage
# ============================================================================


@dataclass
class Stage1Weights:
    """Stage 1: Basic Flight - navigate to target and hover."""
    w_pos: float = 4.0
    a_pos: float = 1.2
    w_vel: float = 1.0
    a_vel: float = 2.0
    w_level: float = 1.0
    a_level: float = 2.0
    w_smooth: float = 0.5
    a_smooth: float = 1.0
    w_mag: float = 0.1
    a_mag: float = 0.5
    alive: float = 0.05


@dataclass
class Stage2Weights:
    """Stage 2: Precision Approach - descend vertically with tight alignment."""
    w_align: float = 4.0
    a_align: float = 1.5
    w_alt: float = 2.0
    a_alt: float = 2.0
    w_desc: float = 1.5
    a_desc: float = 1.5
    w_level: float = 2.0
    a_level: float = 3.0
    w_slow_xy: float = 1.0
    a_slow_xy: float = 3.0
    w_smooth: float = 0.5
    a_smooth: float = 1.0
    w_mag: float = 0.1
    a_mag: float = 0.5
    alive: float = 0.05


@dataclass
class Stage3Weights:
    """Stage 3: Grasping - approach, descend, close gripper."""
    # Inherits Stage 2 for approach + adds grasping terms
    approach: Stage2Weights = field(default_factory=Stage2Weights)
    w_gripper_open: float = 0.3
    r_grasp_bonus: float = 10.0
    w_hold: float = 2.0
    r_drop_penalty: float = -10.0


@dataclass
class Stage4Weights:
    """Stage 4: Loaded Flight - fly to delivery with payload."""
    w_pos: float = 4.0
    a_pos: float = 1.0
    w_stable: float = 2.5
    a_stable: float = 2.0
    w_level: float = 2.0
    a_level: float = 3.0
    w_hold: float = 1.0
    w_smooth: float = 1.0
    a_smooth: float = 1.0
    w_gentle: float = 0.5
    a_gentle: float = 2.0
    alive: float = 0.05


@dataclass
class Stage5Weights:
    """Stage 5: Release - approach delivery, open gripper, ascend."""
    approach: Stage2Weights = field(default_factory=Stage2Weights)
    w_gripper_close: float = 0.3
    r_release_bonus: float = 10.0
    w_ascend: float = 1.5
    a_ascend: float = 1.5


# ============================================================================
# Core reward primitives
# ============================================================================


def exp_reward(error: torch.Tensor, weight: float, alpha: float) -> torch.Tensor:
    """Negative exponential kernel: w * exp(-alpha * error)."""
    return weight * torch.exp(-alpha * error)


def tilt_angle(rot_matrix: torch.Tensor) -> torch.Tensor:
    """Compute tilt angle from rotation matrix.

    Uses 1 - R[2,2] instead of arccos(R[2,2]) to avoid gradient singularity.

    Args:
        rot_matrix: (N, 3, 3) rotation matrices

    Returns:
        tilt: (N,) tilt angles in [0, 2] (0 = level, 2 = inverted)
    """
    return 1.0 - rot_matrix[:, 2, 2]


# ============================================================================
# Stage 1: Basic flight
# ============================================================================


def compute_stage1_rewards(
    pos_w: torch.Tensor,          # (N, 3) body position world
    vel_b: torch.Tensor,          # (N, 3) body linear velocity in body frame
    rot_matrix: torch.Tensor,     # (N, 3, 3) rotation matrix
    goal_w: torch.Tensor,         # (N, 3) goal position world
    action: torch.Tensor,         # (N, A) current action
    prev_action: torch.Tensor,    # (N, A) previous action
    w: Stage1Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 1 rewards. Returns total reward and info dict."""
    w = w or Stage1Weights()

    pos_err = torch.norm(pos_w - goal_w, dim=-1)
    vel_norm = torch.norm(vel_b, dim=-1)

    # Position reward
    r_pos = exp_reward(pos_err, w.w_pos, w.a_pos)

    # Velocity penalty (proximity-gated)
    proximity = (1.0 - pos_err).clamp(min=0.0)
    r_vel = exp_reward(vel_norm * proximity, w.w_vel, w.a_vel)

    # Level flight reward (using 1 - R[2,2] instead of arccos for gradient stability)
    tilt = tilt_angle(rot_matrix)
    r_level = exp_reward(tilt, w.w_level, w.a_level)

    # Action smoothness
    action_diff = torch.norm(action - prev_action, dim=-1)
    r_smooth = exp_reward(action_diff ** 2, w.w_smooth, w.a_smooth)

    # Action magnitude
    action_norm = torch.norm(action, dim=-1)
    r_mag = exp_reward(action_norm ** 2, w.w_mag, w.a_mag)

    # Alive bonus
    r_alive = torch.full_like(r_pos, w.alive)

    total = r_pos + r_vel + r_level + r_smooth + r_mag + r_alive

    info = {
        "r_pos": r_pos, "r_vel": r_vel, "r_level": r_level,
        "r_smooth": r_smooth, "r_mag": r_mag, "pos_error": pos_err,
        "tilt": tilt,
    }
    return total, info


# ============================================================================
# Stage 2: Precision approach
# ============================================================================


def compute_stage2_rewards(
    pos_w: torch.Tensor,           # (N, 3)
    vel_w: torch.Tensor,           # (N, 3) world-frame velocity
    rot_matrix: torch.Tensor,      # (N, 3, 3)
    goal_w: torch.Tensor,          # (N, 3) target position (at current descent altitude)
    target_vz: torch.Tensor,       # (N,) desired descent velocity (negative)
    action: torch.Tensor,
    prev_action: torch.Tensor,
    w: Stage2Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 2 rewards."""
    w = w or Stage2Weights()

    xy_err = torch.norm(pos_w[:, :2] - goal_w[:, :2], dim=-1)
    alt_err = torch.abs(pos_w[:, 2] - goal_w[:, 2])

    # Horizontal alignment
    r_align = exp_reward(xy_err, w.w_align, w.a_align)

    # Altitude control
    r_alt = exp_reward(alt_err, w.w_alt, w.a_alt)

    # Descent speed regulation (only active when aligned)
    is_aligned = (xy_err < 0.1).float()
    vz_err = torch.abs(vel_w[:, 2] - target_vz)
    r_desc = exp_reward(vz_err, w.w_desc, w.a_desc) * is_aligned

    # Level attitude (strengthened)
    tilt = tilt_angle(rot_matrix)
    r_level = exp_reward(tilt, w.w_level, w.a_level)

    # Horizontal speed suppression (inversely proportional to distance)
    v_xy = torch.norm(vel_w[:, :2], dim=-1)
    d_xy = xy_err.clamp(min=0.05)
    r_slow_xy = exp_reward(v_xy / d_xy, w.w_slow_xy, w.a_slow_xy)

    # Smoothness and magnitude (same as stage 1)
    action_diff = torch.norm(action - prev_action, dim=-1)
    r_smooth = exp_reward(action_diff ** 2, w.w_smooth, w.a_smooth)
    r_mag = exp_reward(torch.norm(action, dim=-1) ** 2, w.w_mag, w.a_mag)

    r_alive = torch.full_like(r_align, w.alive)

    total = r_align + r_alt + r_desc + r_level + r_slow_xy + r_smooth + r_mag + r_alive

    info = {
        "r_align": r_align, "r_alt": r_alt, "r_desc": r_desc,
        "r_level": r_level, "r_slow_xy": r_slow_xy, "xy_err": xy_err,
        "alt_err": alt_err, "tilt": tilt,
    }
    return total, info


# ============================================================================
# Stage 3: Grasping
# ============================================================================


def compute_stage3_rewards(
    pos_w: torch.Tensor,
    vel_w: torch.Tensor,
    rot_matrix: torch.Tensor,
    obj_pos_w: torch.Tensor,       # (N, 3) object position world
    plate_angle: torch.Tensor,     # (N,) normalized plate angle [-1, 1]
    is_grasped: torch.Tensor,      # (N,) bool: object currently grasped
    was_grasped: torch.Tensor,     # (N,) bool: was grasped before (for drop detection)
    just_grasped: torch.Tensor,    # (N,) bool: grasped this step (for one-time bonus)
    just_dropped: torch.Tensor,    # (N,) bool: dropped this step
    action: torch.Tensor,
    prev_action: torch.Tensor,
    w: Stage3Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 3 rewards."""
    w = w or Stage3Weights()

    # Compute approach-altitude goal: directly above object
    approach_goal = obj_pos_w.clone()
    approach_goal[:, 2] += 0.15  # 15cm above object for final approach

    # Approach rewards (reuse stage 2 structure)
    target_vz = torch.full((pos_w.shape[0],), -0.1, device=pos_w.device)
    r_approach, approach_info = compute_stage2_rewards(
        pos_w, vel_w, rot_matrix, approach_goal, target_vz,
        action[:, :7] if action.shape[-1] > 7 else action,
        prev_action[:, :7] if prev_action.shape[-1] > 7 else prev_action,
        w.approach,
    )

    # Gripper preparation: keep open when far from object
    d_obj = torch.norm(pos_w - obj_pos_w, dim=-1)
    far_mask = ((~is_grasped) & (d_obj > 0.05)).float()
    r_gripper_open = w.w_gripper_open * plate_angle.clamp(min=0.0) * far_mask

    # Grasp success bonus (one-time)
    r_grasp = w.r_grasp_bonus * just_grasped.float()

    # Hold reward (per step while grasped)
    r_hold = w.w_hold * is_grasped.float()

    # Drop penalty (one-time)
    r_drop = w.r_drop_penalty * just_dropped.float()

    total = r_approach + r_gripper_open + r_grasp + r_hold + r_drop

    info = {
        **approach_info,
        "r_gripper_open": r_gripper_open, "r_grasp": r_grasp,
        "r_hold": r_hold, "r_drop": r_drop, "d_obj": d_obj,
    }
    return total, info


# ============================================================================
# Stage 4: Loaded flight
# ============================================================================


def compute_stage4_rewards(
    pos_w: torch.Tensor,
    vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    rot_matrix: torch.Tensor,
    goal_w: torch.Tensor,
    is_grasped: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    w: Stage4Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 4 rewards."""
    w = w or Stage4Weights()

    pos_err = torch.norm(pos_w - goal_w, dim=-1)
    ang_vel_norm = torch.norm(ang_vel_b, dim=-1)
    tilt = tilt_angle(rot_matrix)
    action_diff = torch.norm(action - prev_action, dim=-1)
    accel_norm = torch.norm(action[:, :3], dim=-1)

    r_pos = exp_reward(pos_err, w.w_pos, w.a_pos)
    r_stable = exp_reward(ang_vel_norm, w.w_stable, w.a_stable)
    r_level = exp_reward(tilt, w.w_level, w.a_level)
    r_hold = w.w_hold * is_grasped.float()
    r_smooth = exp_reward(action_diff ** 2, w.w_smooth, w.a_smooth)
    r_gentle = exp_reward(accel_norm ** 2, w.w_gentle, w.a_gentle)
    r_alive = torch.full_like(r_pos, w.alive)

    total = r_pos + r_stable + r_level + r_hold + r_smooth + r_gentle + r_alive

    info = {
        "r_pos": r_pos, "r_stable": r_stable, "r_level": r_level,
        "r_hold": r_hold, "pos_error": pos_err,
    }
    return total, info


# ============================================================================
# Stage 5: Release
# ============================================================================


def compute_stage5_rewards(
    pos_w: torch.Tensor,
    vel_w: torch.Tensor,
    rot_matrix: torch.Tensor,
    delivery_pos_w: torch.Tensor,
    plate_angle: torch.Tensor,
    is_grasped: torch.Tensor,
    just_released: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    w: Stage5Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 5 rewards."""
    w = w or Stage5Weights()

    # Approach delivery point (same as stage 2)
    target_vz = torch.full((pos_w.shape[0],), -0.1, device=pos_w.device)
    r_approach, approach_info = compute_stage2_rewards(
        pos_w, vel_w, rot_matrix, delivery_pos_w, target_vz,
        action[:, :7] if action.shape[-1] > 7 else action,
        prev_action[:, :7] if prev_action.shape[-1] > 7 else prev_action,
        w.approach,
    )

    # Encourage gripper closing (keeping payload) while approaching
    r_gripper_close = w.w_gripper_close * (1.0 - plate_angle.clamp(min=0.0)) * is_grasped.float()

    # Release bonus (one-time, at correct position)
    xy_err = torch.norm(pos_w[:, :2] - delivery_pos_w[:, :2], dim=-1)
    at_delivery = (xy_err < 0.1).float()
    r_release = w.r_release_bonus * just_released.float() * at_delivery

    # Post-release ascend reward
    released = (~is_grasped).float()
    ascend_vel = vel_w[:, 2].clamp(min=0.0)
    r_ascend = exp_reward(-ascend_vel, w.w_ascend, w.a_ascend) * released

    total = r_approach + r_gripper_close + r_release + r_ascend

    info = {
        **approach_info, "r_release": r_release, "r_ascend": r_ascend,
    }
    return total, info
