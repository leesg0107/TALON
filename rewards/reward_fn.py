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
    """Stage 1: Basic Flight - reach goal and hover.

    Design: r_pos (always positive, gradient) + arrival/hover bonus (big reward at goal).
    No free rewards. Only way to maximize: reach goal and stay.
    """
    w_pos: float = 4.0
    a_pos: float = 1.2
    arrival_threshold: float = 0.5
    arrival_bonus: float = 10.0
    hover_bonus: float = 5.0
    a_hover_vel: float = 3.0
    w_smooth: float = 0.1
    w_mag: float = 0.02


@dataclass
class Stage2Weights:
    """Stage 2: Precision Approach - descend to object from above."""
    # Horizontal alignment (two-scale: coarse for approach, fine for precision)
    w_align: float = 4.0
    a_align: float = 1.5
    w_align_fine: float = 25.0   # ultra-fine: dominate all other rewards for precision
    a_align_fine: float = 15.0   # very sharp: 0.08m → 0.30, 0.04m → 0.55
    # Altitude: exp (precision) + linear (far-field gradient)
    w_alt: float = 2.0
    a_alt: float = 2.0
    w_alt_linear: float = 0.5   # -w * alt_err (constant gradient for descent)
    # Descent speed (gated by horizontal alignment)
    w_desc: float = 1.5
    a_desc: float = 1.5
    # Level attitude (reduced: allow tilting for XY correction, 2.0→1.0)
    w_level: float = 1.0
    a_level: float = 3.0
    # Horizontal speed suppression
    w_slow_xy: float = 1.0
    a_slow_xy: float = 3.0
    # Smoothness: proportional penalty (exp vanishes, Stage 1 lesson)
    w_smooth: float = 0.1       # -w * ||Δa||²
    w_mag: float = 0.02         # -w * ||a||²
    alive: float = 0.05


@dataclass
class Stage3Weights:
    """Stage 3: Grasping - approach, descend, close gripper (physics-based).

    Gripper reward: continuous target tracking (no open/close discontinuity).
    target = open when far, closed when near. Proportional error (not exp).
    """
    approach: Stage2Weights = field(default_factory=Stage2Weights)
    # Gripper target tracking: weight = base + (near - base) * proximity
    w_gripper_track_base: float = 0.5   # gripper reward weight at d >= 0.5m
    w_gripper_track_near: float = 2.0   # gripper reward weight at d = 0m
    gripper_target_dist: float = 0.4    # d at which target transitions fully open→closed
    # Grasp rewards
    r_grasp_bonus: float = 10.0
    w_hold: float = 12.0   # exceeds not-grasped total (~11) → grasped is more rewarding
    r_drop_penalty: float = -50.0  # strong deterrent against dropping


@dataclass
class Stage4Weights:
    """Stage 4: Loaded Flight with goal respawn.

    Key design principle: ALL position-related rewards are ≤ 0 (shifted exp).
    The ONLY positive reward is arrival bonus. This prevents hovering exploit
    (sitting near goal forever) while maintaining dense gradient toward goal.

    Reward ordering must satisfy: arrive > crash > hover-forever
    - arrive: positive (bonuses exceed time cost)
    - crash: zero (terminated, V=0)
    - hover: negative (time cost with no arrivals)
    """
    # Time penalty: makes every step cost → drives toward arrival
    time_penalty: float = -0.5
    # Distance shaping: shifted exp, always ≤ 0, provides gradient toward goal
    w_pos_shape: float = 0.5
    a_pos_shape: float = 1.2
    # Level flight: large enough that survive > crash (+0.6 - 0.5 = +0.1/step)
    w_level: float = 0.6
    a_level: float = 2.0
    # Smoothness: proportional penalties
    w_smooth: float = 0.1
    w_mag: float = 0.02
    # Arrival bonus: only positive goal-seeking reward
    arrival_bonus: float = 200.0


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
    gripper_pos_w: torch.Tensor,  # (N, 3) gripper center position world
    vel_b: torch.Tensor,          # (N, 3) gripper velocity in body frame
    rot_matrix: torch.Tensor,     # (N, 3, 3) rotation matrix
    goal_w: torch.Tensor,         # (N, 3) goal position world
    action: torch.Tensor,         # (N, A) current action
    prev_action: torch.Tensor,    # (N, A) previous action
    w: Stage1Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 1 rewards: reach goal and hover.

    r_pos: always positive, gradient toward goal (no crash exploit).
    r_arrive + r_hover: big bonus at goal → dominates r_pos.
    No free rewards (r_vel, r_level, r_alive removed).
    """
    w = w or Stage1Weights()

    pos_err = torch.norm(gripper_pos_w - goal_w, dim=-1)
    vel_norm = torch.norm(vel_b, dim=-1)

    # Position: always positive, gradient toward goal
    r_pos = exp_reward(pos_err, w.w_pos, w.a_pos)

    # Arrival bonus: big reward when at goal
    arrived = (pos_err < w.arrival_threshold).float()
    r_arrive = w.arrival_bonus * arrived

    # Hover bonus: reward for being still at goal
    r_hover = w.hover_bonus * arrived * torch.exp(-w.a_hover_vel * vel_norm)

    # Action penalties
    action_diff_sq = torch.sum((action - prev_action) ** 2, dim=-1)
    r_smooth = -w.w_smooth * action_diff_sq
    action_sq = torch.sum(action ** 2, dim=-1)
    r_mag = -w.w_mag * action_sq

    total = r_pos + r_arrive + r_hover + r_smooth + r_mag

    info = {
        "r_pos": r_pos, "r_arrive": r_arrive, "r_hover": r_hover,
        "r_smooth": r_smooth, "r_mag": r_mag, "pos_error": pos_err,
        "tilt": tilt_angle(rot_matrix), "arrived": arrived,
    }
    return total, info


# ============================================================================
# Stage 2: Precision approach
# ============================================================================


def compute_stage2_rewards(
    pos_w: torch.Tensor,           # (N, 3) body position world
    vel_w: torch.Tensor,           # (N, 3) world-frame velocity
    rot_matrix: torch.Tensor,      # (N, 3, 3)
    gripper_pos_w: torch.Tensor,   # (N, 3) gripper center position world
    obj_pos_w: torch.Tensor,       # (N, 3) object position world
    action: torch.Tensor,
    prev_action: torch.Tensor,
    w: Stage2Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 2 rewards: gripper-centric precision approach.

    Key change: all alignment rewards measure GRIPPER-to-OBJECT distance,
    not body-to-goal. The policy learns to position the gripper precisely,
    accounting for tilt and moment arm automatically.

    Reward structure:
    - r_align: gripper XY alignment over object (two-scale)
    - r_alt + r_alt_linear: gripper altitude to object altitude
    - r_desc: descent speed tracking, gated by XY alignment
    - r_level: level attitude (reduced weight, allows tilting for XY correction)
    - r_slow_xy: horizontal speed suppression near target
    - r_smooth, r_mag: proportional penalties
    """
    w = w or Stage2Weights()

    # Gripper-centric errors: measure gripper position relative to object
    xy_err = torch.norm(gripper_pos_w[:, :2] - obj_pos_w[:, :2], dim=-1)

    # Horizontal alignment: two-scale kernel (gripper XY over object)
    r_align = exp_reward(xy_err, w.w_align, w.a_align) + exp_reward(xy_err, w.w_align_fine, w.a_align_fine)

    # --- Staged altitude: XY first, then descend ---
    # When XY not aligned: target = staging height (0.3m above object)
    # When XY aligned: target = object height (descend to grasp)
    staging_offset = 0.30
    xy_aligned = (xy_err < 0.10).float()
    alt_target_z = obj_pos_w[:, 2] + staging_offset * (1.0 - xy_aligned)  # obj+0.3 or obj+0.0
    alt_err = torch.abs(gripper_pos_w[:, 2] - alt_target_z)

    # Altitude: gripper height relative to staged target
    r_alt = exp_reward(alt_err, w.w_alt, w.a_alt)
    r_alt_linear = -w.w_alt_linear * alt_err

    # BELOW-TARGET PENALTY: gripper should not go below current target
    below_target = (alt_target_z - gripper_pos_w[:, 2]).clamp(min=0.0)
    r_below = -5.0 * below_target

    # Descent speed: only when XY aligned (descending phase)
    desired_vz = -0.5 * (alt_err / 2.0).clamp(min=0.1, max=1.0)
    is_aligned = xy_aligned
    vz_err = torch.abs(vel_w[:, 2] - desired_vz)
    r_desc = exp_reward(vz_err, w.w_desc, w.a_desc) * is_aligned

    # Level attitude
    tilt = tilt_angle(rot_matrix)
    r_level = exp_reward(tilt, w.w_level, w.a_level)

    # Horizontal speed suppression near target
    v_xy = torch.norm(vel_w[:, :2], dim=-1)
    d_xy = xy_err.clamp(min=0.05)
    slow_gate = (1.0 - ((d_xy - 0.1) / 0.2).clamp(0.0, 1.0))
    r_slow_xy = exp_reward(v_xy / d_xy, w.w_slow_xy, w.a_slow_xy) * slow_gate

    # Smoothness: proportional penalty
    action_diff_sq = torch.sum((action - prev_action) ** 2, dim=-1)
    r_smooth = -w.w_smooth * action_diff_sq

    # Action magnitude: proportional penalty
    action_sq = torch.sum(action ** 2, dim=-1)
    r_mag = -w.w_mag * action_sq

    r_alive = torch.full_like(r_align, w.alive)

    total = r_align + r_alt + r_alt_linear + r_below + r_desc + r_level + r_slow_xy + r_smooth + r_mag + r_alive

    info = {
        "r_align": r_align, "r_alt": r_alt, "r_alt_linear": r_alt_linear,
        "r_below": r_below, "r_desc": r_desc, "r_level": r_level,
        "r_slow_xy": r_slow_xy, "r_smooth": r_smooth, "r_mag": r_mag,
        "xy_err": xy_err, "alt_err": alt_err, "tilt": tilt,
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
    gripper_pos_w: torch.Tensor,   # (N, 3) gripper center position world
    plate_angle: torch.Tensor,     # (N,) normalized plate angle [-1, 1]
    is_grasped: torch.Tensor,      # (N,) bool: object currently grasped
    was_grasped: torch.Tensor,     # (N,) bool: was grasped before (for drop detection)
    just_grasped: torch.Tensor,    # (N,) bool: grasped this step (for one-time bonus)
    just_dropped: torch.Tensor,    # (N,) bool: dropped this step
    action: torch.Tensor,
    prev_action: torch.Tensor,
    w: Stage3Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute unified approach + grasp rewards (used for Stage 2 and 3)."""
    w = w or Stage3Weights()

    # Gripper-centric approach rewards: gripper→object directly (no +0.15m offset)
    r_approach, approach_info = compute_stage2_rewards(
        pos_w, vel_w, rot_matrix, gripper_pos_w, obj_pos_w,
        action[:, :7] if action.shape[-1] > 7 else action,
        prev_action[:, :7] if prev_action.shape[-1] > 7 else prev_action,
        w.approach,
    )

    # Distance from gripper to object
    d_obj = torch.norm(gripper_pos_w - obj_pos_w, dim=-1)

    # --- Continuous gripper target tracking reward ---
    # Target plate angle: open when far, closed when near (linear interpolation)
    #   d >= 0.4m → target = +1.0 (fully open)
    #   d = 0.0m  → target = -1.0 (fully closed)
    alpha = (d_obj / w.gripper_target_dist).clamp(0.0, 1.0)
    target_plate = 2.0 * alpha - 1.0

    # Tracking error (proportional, NOT exp — avoids gradient vanishing)
    angle_err = torch.abs(plate_angle - target_plate)

    # Proximity-scaled weight: gentle when far, strong when near
    proximity01 = (1.0 - d_obj / 0.5).clamp(0.0, 1.0)
    w_gripper = w.w_gripper_track_base + (w.w_gripper_track_near - w.w_gripper_track_base) * proximity01

    # Reward: w * (1 - err). Perfect tracking → +w. Max mismatch (err=2) → -w.
    # Always active (not gated by is_grasped): provides dense gripper guidance
    # Not grasped: "open when far, close when near"
    # Grasped: "stay closed" (d_obj small → target = closed)
    r_gripper_track = w_gripper * (1.0 - angle_err)

    # Grasp success bonus (one-time)
    r_grasp = w.r_grasp_bonus * just_grasped.float()

    # Hold reward (per step while grasped)
    r_hold = w.w_hold * is_grasped.float()

    # Drop penalty (one-time)
    r_drop = w.r_drop_penalty * just_dropped.float()

    total = r_approach + r_gripper_track + r_grasp + r_hold + r_drop

    # Gripper XY offset for monitoring (not used in reward)
    gripper_xy_offset = torch.norm(gripper_pos_w[:, :2] - obj_pos_w[:, :2], dim=-1)

    info = {
        **approach_info,
        "r_gripper_track": r_gripper_track,
        "gripper_xy_offset": gripper_xy_offset, "target_plate": target_plate,
        "r_grasp": r_grasp, "r_hold": r_hold, "r_drop": r_drop, "d_obj": d_obj,
    }
    return total, info


# ============================================================================
# Stage 4: Loaded flight
# ============================================================================


def compute_stage4_rewards(
    gripper_pos_w: torch.Tensor,   # (N, 3) gripper center position world
    rot_matrix: torch.Tensor,      # (N, 3, 3) rotation matrix
    goal_w: torch.Tensor,          # (N, 3) goal position world
    action: torch.Tensor,          # (N, A) current action
    prev_action: torch.Tensor,     # (N, A) previous action
    w: Stage4Weights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Stage 4 rewards: loaded flight with goal respawn.

    All position rewards ≤ 0 (shifted exp). Only arrival bonus is positive.
    Ensures: arrive > crash > hover-forever.
    """
    w = w or Stage4Weights()

    pos_err = torch.norm(gripper_pos_w - goal_w, dim=-1)

    # Time penalty: constant cost per step → drives toward faster arrival
    r_time = torch.full_like(pos_err, w.time_penalty)

    # Distance shaping: shifted exp kernel, always ≤ 0
    # d=0 → 0, d=1.5 → -0.42. Provides gradient without free positive reward.
    r_pos_shape = w.w_pos_shape * (torch.exp(-w.a_pos_shape * pos_err) - 1.0)

    # Level flight: large enough that (r_time + r_level) > 0 → survive > crash
    tilt = tilt_angle(rot_matrix)
    r_level = exp_reward(tilt, w.w_level, w.a_level)

    # Action smoothness: proportional penalty
    action_diff_sq = torch.sum((action - prev_action) ** 2, dim=-1)
    r_smooth = -w.w_smooth * action_diff_sq

    # Action magnitude: proportional penalty
    action_sq = torch.sum(action ** 2, dim=-1)
    r_mag = -w.w_mag * action_sq

    # Note: arrival bonus (+200) is applied in drone_env.py, not here
    total = r_time + r_pos_shape + r_level + r_smooth + r_mag

    info = {
        "r_time": r_time, "r_pos_shape": r_pos_shape, "r_level": r_level,
        "r_smooth": r_smooth, "r_mag": r_mag,
        "pos_error": pos_err, "tilt": tilt,
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
