"""
Waypoint drone environment configuration.

Simplified env for Stage 1 (approach) and Stage 4 (loaded flight).
Separate from GripperDroneEnvCfg — does NOT modify existing code.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Same URDF as main env
_URDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(_URDF_DIR, "gripper_drone_v5.urdf")


# ============================================================================
# Robot (same as main env — shared URDF)
# ============================================================================

WAYPOINT_DRONE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=False,
        make_instanceable=True,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,
                damping=5.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 3.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "plate_joint_left": 0.785,
            "plate_joint_right": 0.785,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "gripper_plates": ImplicitActuatorCfg(
            joint_names_expr=["plate_joint_left", "plate_joint_right"],
            stiffness=300.0,
            damping=15.0,
            effort_limit=50.0,
            velocity_limit=3.0,
        ),
    },
)


# ============================================================================
# Scene
# ============================================================================

# Box for loaded mode (grasped by gripper)
from isaaclab.assets import RigidObjectCfg

GRASP_BOX_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/GraspBox",
    spawn=sim_utils.CuboidCfg(
        size=(0.08, 0.08, 0.08),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005, rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.92)),  # gripper center at spawn
)


@configclass
class WaypointSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = WAYPOINT_DRONE_CFG


@configclass
class WaypointLoadedSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = WAYPOINT_DRONE_CFG
    grasp_box: RigidObjectCfg = GRASP_BOX_CFG


# ============================================================================
# Domain Randomization
# ============================================================================

@dataclass
class WaypointDomainRandCfg:
    mass_scale_range: tuple[float, float] = (0.9, 1.1)
    motor_k_f_scale: tuple[float, float] = (0.85, 1.15)
    wind_force_std: float = 0.5
    wind_freq: float = 0.5
    pos_noise_std: float = 0.01
    vel_noise_std: float = 0.03


# ============================================================================
# Main Config
# ============================================================================

@configclass
class WaypointEnvCfg(DirectRLEnvCfg):
    """Simplified waypoint-following drone environment."""

    # Mode: "flight" (Stage 1) or "loaded" (Stage 4)
    mode: str = "flight"

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
    scene: WaypointSceneCfg = WaypointSceneCfg(env_spacing=5.0)

    # Terrain
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

    # Control
    decimation: int = 2  # 300Hz sim → 150Hz policy

    # Episode
    episode_length_s: float = 20.0  # 20s for multi-waypoint learning

    # Spaces: 22D obs (flight) or 23D (loaded), 4D action
    # vel_b(3) + ang_vel_b(3) + R_flat(9) + goal_b(3) + prev_action(4) = 22D
    # Stage 4 adds payload_est(1) = 23D
    observation_space: int = 22  # override in __post_init__ for loaded mode
    action_space: int = 4
    state_space: int | None = 0

    # Domain randomization
    domain_rand: WaypointDomainRandCfg = WaypointDomainRandCfg()

    # Goal generation
    goal_range_xy: float = 2.0
    goal_range_z: tuple[float, float] = (1.5, 3.5)
    spawn_z: float = 3.0
    spawn_spread: float = 1.0

    # Stage 4: payload
    payload_mass_range: tuple[float, float] = (0.15, 0.25)

    # Termination
    max_tilt_deg: float = 60.0
    min_altitude: float = 0.15
    max_distance: float = 10.0

    # Obstacle config (for future extension)
    enable_obstacles: bool = False
    num_obstacles: int = 0
    obstacle_radius: float = 0.15
    obstacle_height: float = 3.0

    def __post_init__(self):
        super().__post_init__()
        if self.mode == "loaded":
            self.observation_space = 23  # +1 for payload_est
            self.scene = WaypointLoadedSceneCfg(env_spacing=self.scene.env_spacing)
