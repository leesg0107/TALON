"""
Environment configuration for all Gripper-Drone training stages.

Uses Isaac Lab's @configclass system for structured configuration.
"""

from __future__ import annotations

import os
from enum import IntEnum
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

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
        # CRITICAL: Do NOT merge fixed joints — we need motor bodies as separate
        # rigid bodies to apply per-motor external forces at correct positions.
        merge_fixed_joints=False,
        # Force sensors on gripper plates for contact detection
        activate_contact_sensors=True,
        # Joint drive configuration for URDF revolute joints
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,
                damping=5.0,
            ),
        ),
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
            stiffness=300.0,   # stronger closing force
            damping=15.0,      # stable grip
            effort_limit=50.0,  # high force limit
            velocity_limit=3.0,
        ),
    },
)


# ============================================================================
# Grasp target object
# ============================================================================

GRASP_OBJECT_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.08, 0.08, 0.08),  # 8cm cube (must match training)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,  # Dynamic: box can be grasped and lifted
            disable_gravity=False,    # Gravity ON: box sits on pedestal naturally
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.2),
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.54)),  # on pedestal (top=0.50 + box_half=0.04)
)


# ============================================================================
# Pedestal (table/pillar for box to sit on)
# ============================================================================

PEDESTAL_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Pedestal",
    spawn=sim_utils.CuboidCfg(
        size=(0.30, 0.30, 0.50),  # 30x30cm top, 50cm tall
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # Fixed: doesn't move
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.5),
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),  # center at 0.25m → top at 0.50m
)


# ============================================================================
# Scene configuration
# ============================================================================

@configclass
class GripperDroneSceneCfg(InteractiveSceneCfg):
    """Scene with drone + grasp object + pedestal."""
    num_envs: int = 4096
    env_spacing: float = 5.0

    # Robot
    robot: ArticulationCfg = GRIPPER_DRONE_CFG

    # Grasp target: always present in all stages for consistent 31D observations.
    # Stage 1: on ground (non-interfering). Stage 2+: on pedestal.
    grasp_object: RigidObjectCfg = GRASP_OBJECT_CFG

    # Pedestal: table/pillar for box to sit on (Stage 2+)
    pedestal: RigidObjectCfg = PEDESTAL_CFG


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
            static_friction=2.5,              # moderate friction for gripper grip
            dynamic_friction=2.5,
            restitution=0.0,
        ),
    )

    # Scene
    scene: GripperDroneSceneCfg = GripperDroneSceneCfg()

    # Terrain (ground plane)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Control decimation: sim at 300Hz, policy at 150Hz
    decimation: int = 2

    # Episode
    episode_length_s: float = 6.0

    # Observation / Action spaces: unified 31D/8D for ALL stages.
    # No dimension changes between stages. Gripper locked in Stage 1-2.
    observation_space: int = 31
    action_space: int = 8
    state_space: int | None = 0

    # Domain randomization
    domain_rand: DomainRandCfg = DomainRandCfg()

    # --- Stage-specific task parameters ---

    # Stage 1: Basic flight
    goal_pos_range_xy: float = 1.0       # Goal XY sampled from [-range, +range] (Sun: 1.0)
    goal_pos_range_z: tuple[float, float] = (2.0, 4.0)  # (Sun: 3.0-5.0, adjusted for our spawn)
    goal_change_interval_s: float = 6.0  # New goal every N seconds
    spawn_spread: float = 1.0            # Random spawn offset

    # Stage 2: Precision approach
    approach_start_z: tuple[float, float] = (3.0, 5.0)
    target_z: tuple[float, float] = (0.5, 1.5)
    descent_speed: float = -0.1          # m/s (negative = down)
    hold_time_s: float = 5.0             # Must hover 0.2m above target for this long

    # Stage 3: Grasping
    lock_gripper: bool = True            # Default: locked. Stage 3+ unlocks.
    auto_grasp: bool = False             # Physics-based: policy controls gripper
    auto_grasp_prob: float = 0.0         # No auto-grasp
    grasp_trigger_dist: float = 0.15     # meters (gripper tip to object center)
    grasp_trigger_speed: float = 0.1     # m/s (unused in physics-based mode)
    grasp_trigger_tilt: float = 0.087    # ~5 degrees (unused in physics-based mode)
    grasp_plate_threshold: float = 0.1   # rad: plates must be < this to count as closed
    object_mass_range: tuple[float, float] = (0.05, 0.5)
    object_size_range: tuple[float, float] = (0.03, 0.12)

    # Stage 4: Loaded flight
    delivery_range_xy: float = 3.0
    delivery_z: float = 2.0

    # Stage 5: Release (reuses Stage 2/3 params)

    # Termination
    max_tilt_deg: float = 60.0
    min_altitude: float = 0.10  # lowered: gripping center at -0.08 needs body at ~0.13m
    max_distance: float = 10.0

    def __post_init__(self):
        """Set stage-dependent parameters.

        All stages use 31D obs + 8D action (unified architecture).
        Only reward structure, episode length, and lock_gripper differ.
        """
        super().__post_init__()

        # observation_space=31, action_space=8, grasp_object already set as defaults.
        # Stage-specific: only task parameters and lock_gripper.

        if self.stage == Stage.BASIC_FLIGHT:
            self.lock_gripper = True
            self.domain_rand.payload_mass_range = (0.0, 0.0)

        elif self.stage == Stage.PRECISION_APPROACH:
            self.lock_gripper = True   # Gripper locked open: focus on align
            self.domain_rand.payload_mass_range = (0.0, 0.0)
            self.episode_length_s = 12.0  # Balance: enough time for precise docking

        elif self.stage == Stage.GRASPING:
            self.lock_gripper = True   # Gripper locked open during approach; auto-close on dock
            self.domain_rand.payload_mass_range = (0.0, 0.0)
            self.episode_length_s = 15.0

        elif self.stage == Stage.LOADED_FLIGHT:
            self.lock_gripper = False  # Holding payload
            self.domain_rand.payload_mass_range = (0.05, 0.5)

        elif self.stage == Stage.RELEASE:
            self.lock_gripper = False  # Releasing payload
            self.domain_rand.payload_mass_range = (0.05, 0.5)
