import gymnasium as gym
from .env_cfg import GripperDroneEnvCfg, Stage
from .drone_env import GripperDroneEnv

# Register all stage environments
for stage_id in range(1, 6):
    gym.register(
        id=f"GripperDrone-Stage{stage_id}-v0",
        entry_point="envs.drone_env:GripperDroneEnv",
        disable_env_checker=True,
        kwargs={"cfg": GripperDroneEnvCfg(stage=Stage(stage_id))},
    )
