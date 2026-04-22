"""
PPO agent configuration for WaypointDroneEnv.

Small network (2x128) based on Swift (Nature 2023) architecture.
Separate from ppo_cfg.py — does NOT modify existing agent code.
"""
import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory


# ============================================================================
# Small 2x128 Policy (Swift architecture)
# ============================================================================

class WaypointPolicy(GaussianMixin, Model):
    """Small Gaussian policy for waypoint following. 2x128 MLP."""

    def __init__(self, observation_space, action_space, device,
                 initial_log_std=0.0, min_log_std=-2.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
                                min_log_std=min_log_std, max_log_std=2.0)

        obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space
        act_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), initial_log_std))

        # Orthogonal init (standard for PPO)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std, {}


class WaypointValue(DeterministicMixin, Model):
    """Small value network. 2x128 MLP."""

    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# ============================================================================
# PPO Config
# ============================================================================

def get_waypoint_ppo_config(mode: str = "flight"):
    """PPO hyperparameters for waypoint following.

    Based on Swift (Nature 2023) + Aerial Gym recommendations.
    """
    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"] = 100               # Genesis: 100 steps per env per update
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 4             # Genesis: larger batches (8192*100/batch)
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = None
    cfg["random_timesteps"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": 1, "device": None}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": None}

    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    # No dock phase in loaded mode → same entropy as flight
    cfg["entropy_loss_scale"] = 0.004       # Genesis: 0.004
    cfg["value_loss_scale"] = 1.0
    cfg["grad_norm_clip"] = 1.0
    cfg["kl_threshold"] = 0

    return cfg


def _warm_start_from_flight(agent, flight_ckpt: str, device):
    """Load flight (22D obs) weights into loaded (23D obs) model.

    First linear layer: flight has shape (128, 22), loaded needs (128, 23).
    Copy the 22 columns, init the 23rd (payload_est) to zero.
    All other layers copy directly.
    """
    ckpt = torch.load(flight_ckpt, map_location=device, weights_only=True)

    for model_name in ["policy", "value"]:
        model = agent.models[model_name]
        if model_name not in ckpt:
            continue
        src_sd = ckpt[model_name]
        dst_sd = model.state_dict()

        for key in dst_sd:
            if key in src_sd:
                if src_sd[key].shape == dst_sd[key].shape:
                    dst_sd[key] = src_sd[key]
                elif src_sd[key].dim() == 2 and src_sd[key].shape[0] == dst_sd[key].shape[0]:
                    # First layer weight: (128, 22) → (128, 23)
                    dst_sd[key][:, :src_sd[key].shape[1]] = src_sd[key]
                    print(f"  Warm-start {model_name}.{key}: {src_sd[key].shape} → {dst_sd[key].shape}")

        model.load_state_dict(dst_sd)
        print(f"  Loaded {model_name} weights ({len(src_sd)} params)")

    print(f"  Warm-start from flight checkpoint: {flight_ckpt}")


def build_waypoint_agent(
    env,
    device: torch.device,
    mode: str = "flight",
    checkpoint_path: str | None = None,
    warm_start_flight: str | None = None,
) -> PPO:
    """Build PPO agent for WaypointDroneEnv.

    Args:
        checkpoint_path: Load exact checkpoint (same obs dim required).
        warm_start_flight: Load flight (22D) checkpoint into loaded (23D) model.
    """

    obs_space = env.observation_space
    act_space = env.action_space

    min_log_std = -2.0
    models = {
        "policy": WaypointPolicy(obs_space, act_space, device, min_log_std=min_log_std),
        "value": WaypointValue(obs_space, act_space, device),
    }

    memory = RandomMemory(memory_size=100, num_envs=env.num_envs, device=device)

    cfg = get_waypoint_ppo_config(mode)
    cfg["state_preprocessor_kwargs"]["size"] = obs_space
    cfg["state_preprocessor_kwargs"]["device"] = device
    cfg["value_preprocessor_kwargs"]["size"] = 1
    cfg["value_preprocessor_kwargs"]["device"] = device

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    if checkpoint_path:
        agent.load(checkpoint_path)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    elif warm_start_flight:
        _warm_start_from_flight(agent, warm_start_flight, device)

    return agent
