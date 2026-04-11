"""
SKRL SAC agent configuration and model definitions.

Network architecture matches PPO: 512 -> 256 -> 128 (ELU activation).
SAC requires: policy (Gaussian), critic_1, critic_2, target_critic_1, target_critic_2.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Any

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory


# ============================================================================
# Policy network (Actor) — same architecture as PPO policy
# ============================================================================


class SACPolicy(GaussianMixin, Model):
    """Gaussian policy for SAC.

    Architecture: obs -> 512 -> 256 -> 128 -> action_dim
    Activation: ELU throughout
    Output: tanh-squashed actions in [-1, 1]

    SAC uses log_std as network output (not learnable parameter),
    but SKRL's GaussianMixin handles this via clip_log_std.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
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

        # Orthogonal init (same as PPO)
        for m in self.net:
            if isinstance(m, nn.Linear):
                if m is self.net[-2]:  # output layer (before Tanh)
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def compute(self, inputs: dict, role: str = "") -> dict:
        obs = inputs["states"]
        mean = self.net(obs)
        return mean, self.log_std_parameter, {}


# ============================================================================
# Critic network (Q-function) — takes (state, action) → Q-value
# ============================================================================


class SACCritic(DeterministicMixin, Model):
    """Q-function for SAC. Takes (state, action) as input.

    Architecture: (obs + action) -> 512 -> 256 -> 128 -> 1
    """

    def __init__(self, observation_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        obs_dim = self.num_observations
        act_dim = self.num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # Orthogonal init
        for m in self.net:
            if isinstance(m, nn.Linear):
                if m is self.net[-1]:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def compute(self, inputs: dict, role: str = "") -> dict:
        obs = inputs["states"]
        actions = inputs["taken_actions"]
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x), {}


# ============================================================================
# SAC Agent builder
# ============================================================================


def get_sac_config(stage: int = 1) -> dict:
    """Get SAC hyperparameters for a given training stage."""
    cfg = SAC_DEFAULT_CONFIG.copy()

    # Based on Raffin (2026) SAC + parallel sim tuning + drone control papers
    cfg["gradient_steps"] = 16           # replay ratio = 16/512 = 0.031 (balance speed/quality)
    cfg["batch_size"] = 512
    cfg["discount_factor"] = 0.983       # Raffin: slightly lower for parallel sim
    cfg["polyak"] = 0.002                # Raffin: slower target update
    cfg["actor_learning_rate"] = 4.5e-4
    cfg["critic_learning_rate"] = 4.5e-4
    cfg["learning_rate_scheduler"] = None
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": 1, "device": None}
    cfg["random_timesteps"] = 2000       # random exploration
    cfg["learning_starts"] = 5000        # Q-network warm-up
    cfg["grad_norm_clip"] = 1.0

    # Entropy: standard SAC auto-tuning
    cfg["learn_entropy"] = True
    cfg["entropy_learning_rate"] = 1e-4
    cfg["initial_entropy_value"] = 0.002  # matches v3 converged value + entropy floor
    cfg["target_entropy"] = None         # auto: -dim(A) = -8.0 (standard formula)

    # Stage-specific
    if stage >= 3:
        cfg["actor_learning_rate"] = 3e-4
        cfg["critic_learning_rate"] = 3e-4

    return cfg


def build_sac_agent(
    env,
    device: torch.device,
    stage: int = 1,
    checkpoint_path: str | None = None,
    ppo_policy_path: str | None = None,
    memory_size: int = 10_000,
) -> SAC:
    """Build and optionally load a SAC agent.

    Args:
        env: Isaac Lab environment (wrapped for SKRL)
        device: torch device
        stage: training stage (1-5)
        checkpoint_path: optional SAC checkpoint to load (all models)
        ppo_policy_path: optional PPO checkpoint to load policy weights only
                         (critics initialized fresh — for PPO→SAC transfer)
        memory_size: replay buffer size (per env)

    Returns:
        Configured SAC agent
    """
    obs_space = env.observation_space
    act_space = env.action_space

    # Build models: policy + 2 critics + 2 target critics
    models = {
        "policy": SACPolicy(obs_space, act_space, device),
        "critic_1": SACCritic(obs_space, act_space, device),
        "critic_2": SACCritic(obs_space, act_space, device),
        "target_critic_1": SACCritic(obs_space, act_space, device),
        "target_critic_2": SACCritic(obs_space, act_space, device),
    }

    # Replay buffer (much larger than PPO's rollout buffer)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)

    # Config
    cfg = get_sac_config(stage)
    cfg["state_preprocessor_kwargs"]["size"] = obs_space
    cfg["state_preprocessor_kwargs"]["device"] = device

    # Build agent
    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    # Load checkpoint
    if checkpoint_path is not None:
        agent.load(checkpoint_path)
        print(f"[SAC] Loaded full checkpoint: {checkpoint_path}")
    elif ppo_policy_path is not None:
        # Transfer PPO policy weights to SAC policy (critics stay fresh)
        ppo_state = torch.load(ppo_policy_path, map_location=device)
        # SKRL saves as {"policy": state_dict, "value": state_dict, ...}
        ppo_policy_state = None
        if "policy" in ppo_state:
            ppo_policy_state = ppo_state["policy"]
        elif "models" in ppo_state and "policy" in ppo_state["models"]:
            ppo_policy_state = ppo_state["models"]["policy"]
        else:
            # Try loading the whole dict as state_dict
            for key in ppo_state:
                if "net" in str(ppo_state[key]) or "log_std" in key:
                    ppo_policy_state = ppo_state
                    break

        if ppo_policy_state is not None:
            models["policy"].load_state_dict(ppo_policy_state, strict=True)
            # Reset log_std to 0.0 — PPO's converged low std causes entropy collapse in SAC
            with torch.no_grad():
                models["policy"].log_std_parameter.fill_(0.0)
            print(f"[SAC] Loaded PPO policy weights from: {ppo_policy_path}")
            print(f"[SAC] Reset log_std to 0.0 (was: PPO converged values)")
            print(f"[SAC] Critics initialized fresh (no PPO value function)")
        else:
            print(f"[SAC] WARNING: Could not find policy weights in {ppo_policy_path}")

        # Transfer state_preprocessor (obs normalization) if available
        if "state_preprocessor" in ppo_state:
            agent._state_preprocessor.load_state_dict(ppo_state["state_preprocessor"])
            print(f"[SAC] Loaded state_preprocessor from PPO checkpoint")

    return agent
