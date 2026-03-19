"""
Environment configuration for all Gripper-Drone training stages.

Uses Isaac Lab's @configclass system for structured configuration.
"""

from __future__ import annotations

import os
from enum import IntEnum
from dataclasses import dataclass

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.utils import configclass

# Path to the URDF (relative to this file's directory)
_URDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(_URDF_DIR, "gripper_drone_v5.urdf")


class Stage(IntEnum):
    BASIC_FLIGHT = 1
    PRECISION_APPROACH = 2
    GRASPING = 3
    LOADED_FLIGHT = 4
    RELEASE = 5


# ============================================================================
# Robot configuration
# ============================================================================

GRIPPER_DRONE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        make_instanceable=True,
        # Force sensors on gripper plates for contact detection
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 3.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz identity
        joint_pos={
            "plate_joint_left": 0.785,    # 45 deg (landing configuration)
            "plate_joint_right": 0.785,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "gripper_plates": ImplicitActuatorCfg(
            joint_names_expr=["plate_joint_left", "plate_joint_right"],
            stiffness=40.0,
            damping=5.0,
            effort_limit=8.0,
            velocity_limit=3.0,
        ),
    },
)


# ============================================================================
# Scene configuration
# ============================================================================

@configclass
class GripperDroneSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane and drone."""
    num_envs: int = 4096
    env_spacing: float = 5.0

    # Ground plane
    ground = sim_utils.GroundPlaneCfg()

    # Robot
    robot: ArticulationCfg = GRIPPER_DRONE_CFG


# ============================================================================
# Domain randomization parameters
# ============================================================================

@dataclass
class DomainRandCfg:
    """Domain randomization ranges. Applied per-episode at reset."""
    # Mass randomization (multiplicative factor)
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    # Payload mass [kg] (added to base_link)
    payload_mass_range: tuple[float, float] = (0.0, 0.0)  # stage-dependent
    # Motor constant randomization (multiplicative factor)
    motor_k_f_scale: tuple[float, float] = (0.85, 1.15)
    # CoM offset [m]
    com_offset_range: float = 0.005
    # Wind force [N] per axis (Gaussian std)
    wind_force_std: float = 0.5
    # Wind variation frequency [Hz]
    wind_freq: float = 0.5
    # Sensor noise std
    pos_noise_std: float = 0.02
    vel_noise_std: float = 0.05
    # Plate joint friction multiplier
    plate_friction_range: tuple[float, float] = (0.0, 1.5)
    # Object detection noise (stage 3+)
    obj_detection_noise_std: float = 0.03
    detection_delay_frames: int = 3


# ============================================================================
# Per-stage environment config
# ============================================================================

@configclass
class GripperDroneEnvCfg(DirectRLEnvCfg):
    """Main environment configuration. Stage-dependent parameters are set in __post_init__."""

    # Stage selection
    stage: Stage = Stage.BASIC_FLIGHT

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 300.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Scene
    scene: GripperDroneSceneCfg = GripperDroneSceneCfg()

    # Control decimation: sim at 300Hz, policy at 150Hz
    decimation: int = 2

    # Episode
    episode_length_s: float = 6.0

    # Observation / Action dimensions (set per stage in __post_init__)
    num_observations: int = 23
    num_actions: int = 7
    num_states: int = 0

    # Domain randomization
    domain_rand: DomainRandCfg = DomainRandCfg()

    # --- Stage-specific task parameters ---

    # Stage 1: Basic flight
    goal_pos_range_xy: float = 2.0       # Goal XY sampled from [-range, +range]
    goal_pos_range_z: tuple[float, float] = (1.0, 4.0)
    goal_change_interval_s: float = 6.0  # New goal every N seconds
    spawn_spread: float = 1.0            # Random spawn offset

    # Stage 2: Precision approach
    approach_start_z: tuple[float, float] = (3.0, 5.0)
    target_z: tuple[float, float] = (0.5, 1.5)
    descent_speed: float = -0.1          # m/s (negative = down)
    hold_time_s: float = 5.0             # Must hover 0.2m above target for this long

    # Stage 3: Grasping
    auto_grasp: bool = True              # Stage 3a/3b: auto-close gripper
    auto_grasp_prob: float = 1.0         # Stage 3c: probability of auto-grasp
    grasp_trigger_dist: float = 0.03     # meters
    grasp_trigger_speed: float = 0.1     # m/s
    grasp_trigger_tilt: float = 0.087    # ~5 degrees
    object_mass_range: tuple[float, float] = (0.05, 0.5)
    object_size_range: tuple[float, float] = (0.03, 0.12)

    # Stage 4: Loaded flight
    delivery_range_xy: float = 3.0
    delivery_z: float = 2.0

    # Stage 5: Release (reuses Stage 2/3 params)

    # Termination
    max_tilt_deg: float = 60.0
    min_altitude: float = 0.3
    max_distance: float = 10.0

    def __post_init__(self):
        """Set stage-dependent parameters."""
        super().__post_init__()

        if self.stage == Stage.BASIC_FLIGHT:
            self.num_observations = 23
            self.num_actions = 7
            self.domain_rand.payload_mass_range = (0.0, 0.0)

        elif self.stage == Stage.PRECISION_APPROACH:
            self.num_observations = 23
            self.num_actions = 7
            self.domain_rand.payload_mass_range = (0.0, 0.0)

        elif self.stage == Stage.GRASPING:
            self.num_observations = 31
            self.num_actions = 8  # +1 for gripper command
            self.domain_rand.payload_mass_range = (0.05, 0.5)

        elif self.stage == Stage.LOADED_FLIGHT:
            self.num_observations = 23
            self.num_actions = 7
            self.domain_rand.payload_mass_range = (0.05, 0.5)

        elif self.stage == Stage.RELEASE:
            self.num_observations = 31
            self.num_actions = 8
            self.domain_rand.payload_mass_range = (0.05, 0.5)
