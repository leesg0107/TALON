"""
SKRL PPO agent configuration and model definitions.

Network architecture: 512 -> 256 -> 128 (ELU activation)
Following the design spec and DSAM architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory


# ============================================================================
# Policy network (Actor)
# ============================================================================


class GaussianPolicy(GaussianMixin, Model):
    """Gaussian policy with learnable log-std for PPO.

    Architecture: obs -> 512 -> 256 -> 128 -> action_dim
    Activation: ELU throughout
    Output: tanh-squashed actions in [-1, 1]
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
        max_log_std: float = 1.0,
        initial_log_std: float = 0.0,
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        obs_dim = self.num_observations
        act_dim = self.num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

        self.log_std_parameter = nn.Parameter(
            torch.full((act_dim,), initial_log_std)
        )

        # Initialize weights (orthogonal, PPO best practices)
        # Hidden layers: gain=sqrt(2) for ELU variance preservation
        # Output layer (before Tanh): gain=0.01 for near-zero initial actions
        import math
        for m in self.net:
            if isinstance(m, nn.Linear):
                if m is self.net[-2]:  # output layer (before Tanh)
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:  # hidden layers
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def compute(self, inputs: dict, role: str = "") -> dict:
        obs = inputs["states"]
        mean = self.net(obs)
        return mean, self.log_std_parameter, {}


# ============================================================================
# Value network (Critic)
# ============================================================================


class ValueNetwork(DeterministicMixin, Model):
    """State value function V(s) for PPO.

    Same architecture as policy but scalar output.
    """

    def __init__(self, observation_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = self.num_observations

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # Initialize weights (orthogonal, PPO best practices)
        # Hidden layers: gain=sqrt(2) for ELU, output layer: gain=1.0
        import math
        for m in self.net:
            if isinstance(m, nn.Linear):
                if m is self.net[-1]:  # output layer (scalar value, no activation)
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:  # hidden layers
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def compute(self, inputs: dict, role: str = "") -> dict:
        obs = inputs["states"]
        value = self.net(obs)
        return value, {}


# ============================================================================
# PPO Agent builder
# ============================================================================


def get_ppo_config(stage: int = 1, substage: str | None = None) -> dict:
    """Get PPO hyperparameters for a given training stage.

    All values from the design spec Section 9.
    """
    cfg = PPO_DEFAULT_CONFIG.copy()

    # Hyperparameters for gripper-drone curriculum learning
    # Based on Sun et al. ICRA 2026 (same architecture) + empirical tuning
    cfg["rollouts"] = 24                # n_steps per rollout
    cfg["learning_epochs"] = 5          # PPO epochs per update
    cfg["mini_batches"] = 24            # batch_size = 4096 * 24 / 24 = 4096 per mini-batch
    cfg["discount_factor"] = 0.99       # gamma
    cfg["lambda"] = 0.95                # GAE lambda
    cfg["learning_rate"] = 3e-4         # Adam LR (Sun et al. reference)
    cfg["learning_rate_scheduler"] = None
    cfg["random_timesteps"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": 1, "device": None}  # will be set
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": None}

    # PPO-specific
    cfg["ratio_clip"] = 0.2            # epsilon for clipping
    cfg["value_clip"] = 0.2
    cfg["entropy_loss_scale"] = 0.005  # entropy coefficient (1차 성공 학습 값)
    cfg["value_loss_scale"] = 1.0      # value loss coefficient
    cfg["grad_norm_clip"] = 1.0        # max gradient norm
    cfg["kl_threshold"] = 0            # no KL early stopping (let PPO clip handle it)

    # Stage-specific adjustments
    if stage >= 3:
        cfg["learning_rate"] = 1.5e-4  # Stage 3: standard lr for new task learning
        if substage == "adapt":
            # Stage 2.5: use Stage 2 exploration params (network adaptation only)
            # Higher entropy caused std 0.36→0.52 divergence in first attempt
            pass  # Keep default entropy=0.005, let std converge naturally
        else:
            # Stage 3a+: higher entropy to prevent exploration collapse
            cfg["entropy_loss_scale"] = 0.01
    if stage == 4:
        cfg["learning_rate"] = 1.5e-4

    return cfg


def build_ppo_agent(
    env,
    device: torch.device,
    stage: int = 1,
    substage: str | None = None,
    checkpoint_path: str | None = None,
) -> PPO:
    """Build and optionally load a PPO agent.

    Args:
        env: Isaac Lab environment (wrapped for SKRL)
        device: torch device
        stage: training stage (1-5)
        substage: sub-stage (adapt/a/b/c)
        checkpoint_path: optional path to load weights from previous stage

    Returns:
        Configured PPO agent
    """
    obs_space = env.observation_space
    act_space = env.action_space

    # Build models
    # Stage 3+ (except adapt): higher min_log_std to prevent exploration collapse
    if stage >= 3 and substage != "adapt":
        min_log_std = -2.0
    else:
        min_log_std = -5.0  # Stage 1/2/adapt: standard range
    models = {
        "policy": GaussianPolicy(obs_space, act_space, device,
                                 initial_log_std=0.0, min_log_std=min_log_std),
        "value": ValueNetwork(obs_space, act_space, device),
    }

    # Memory
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

    # Config
    cfg = get_ppo_config(stage, substage=substage)
    cfg["state_preprocessor_kwargs"]["size"] = obs_space
    cfg["state_preprocessor_kwargs"]["device"] = device
    cfg["value_preprocessor_kwargs"]["size"] = 1
    cfg["value_preprocessor_kwargs"]["device"] = device

    # Build agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    # Load checkpoint (unified 31D/8D — no dimension extension needed)
    if checkpoint_path is not None:
        agent.load(checkpoint_path)
        print(f"[PPO] Loaded checkpoint: {checkpoint_path}")

    return agent
