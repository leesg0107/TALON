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
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
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

        # Initialize weights (orthogonal, as in PPO best practices)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        # Last layer smaller gain for initial near-zero actions
        nn.init.orthogonal_(self.net[-2].weight, gain=0.01)

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

        # Initialize weights
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def compute(self, inputs: dict, role: str = "") -> dict:
        obs = inputs["states"]
        value = self.net(obs)
        return value, {}


# ============================================================================
# PPO Agent builder
# ============================================================================


def get_ppo_config(stage: int = 1) -> dict:
    """Get PPO hyperparameters for a given training stage.

    All values from the design spec Section 9.
    """
    cfg = PPO_DEFAULT_CONFIG.copy()

    cfg["rollouts"] = 24                # n_steps per rollout
    cfg["learning_epochs"] = 5          # PPO epochs per update
    cfg["mini_batches"] = 24            # 98304 / 4096 = 24
    cfg["discount_factor"] = 0.99       # gamma
    cfg["lambda"] = 0.95                # GAE lambda
    cfg["learning_rate"] = 3e-4         # Adam LR
    cfg["learning_rate_scheduler"] = None
    cfg["random_timesteps"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": 1, "device": None}  # will be set
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": None}

    # PPO-specific
    cfg["ratio_clip"] = 0.2            # epsilon for clipping
    cfg["value_clip"] = 0.2
    cfg["entropy_loss_scale"] = 0.01   # entropy coefficient
    cfg["value_loss_scale"] = 1.0      # value loss coefficient
    cfg["grad_norm_clip"] = 1.0        # max gradient norm
    cfg["kl_threshold"] = 0            # disable KL early stopping

    # Stage-specific adjustments
    if stage >= 3:
        # Reduce LR for fine-tuning stages
        cfg["learning_rate"] = 1.5e-4
    if stage == 4:
        cfg["learning_rate"] = 1.5e-4

    return cfg


def build_ppo_agent(
    env,
    device: torch.device,
    stage: int = 1,
    checkpoint_path: str | None = None,
) -> PPO:
    """Build and optionally load a PPO agent.

    Args:
        env: Isaac Lab environment (wrapped for SKRL)
        device: torch device
        stage: training stage (1-5)
        checkpoint_path: optional path to load weights from previous stage

    Returns:
        Configured PPO agent
    """
    obs_space = env.observation_space
    act_space = env.action_space

    # Build models
    models = {
        "policy": GaussianPolicy(obs_space, act_space, device, initial_log_std=0.0),
        "value": ValueNetwork(obs_space, act_space, device),
    }

    # Memory
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)

    # Config
    cfg = get_ppo_config(stage)
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

    # Load checkpoint from previous stage if provided
    if checkpoint_path is not None:
        agent.load(checkpoint_path)
        print(f"[PPO] Loaded checkpoint: {checkpoint_path}")

        # If action space changed (e.g., stage 2->3 adds gripper), reinitialize last layer
        if stage == 3 or stage == 5:
            _extend_policy_action_dim(agent, old_dim=7, new_dim=8, device=device)

    return agent


def _extend_policy_action_dim(agent: PPO, old_dim: int, new_dim: int, device: torch.device):
    """Extend the policy output layer when action space grows between stages.

    Copies old weights and initializes new action dimension weights to small random values.
    """
    policy = agent.policy
    # The output layer is net[-2] (before Tanh)
    old_layer = policy.net[-2]
    if old_layer.out_features == old_dim:
        new_layer = nn.Linear(old_layer.in_features, new_dim, device=device)
        # Copy old weights
        with torch.no_grad():
            new_layer.weight[:old_dim] = old_layer.weight
            new_layer.bias[:old_dim] = old_layer.bias
            # Initialize new dimension
            nn.init.orthogonal_(new_layer.weight[old_dim:].unsqueeze(0), gain=0.01)
            new_layer.bias[old_dim:] = 0.0
        policy.net[-2] = new_layer

        # Also extend log_std
        old_log_std = policy.log_std_parameter.data
        new_log_std = torch.zeros(new_dim, device=device)
        new_log_std[:old_dim] = old_log_std
        policy.log_std_parameter = nn.Parameter(new_log_std)

        print(f"[PPO] Extended policy action dim: {old_dim} -> {new_dim}")
