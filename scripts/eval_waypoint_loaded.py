"""Headless eval for Stage 4 loaded flight (physical box in gripper)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

cfg = WaypointEnvCfg(mode="loaded")
cfg.scene.num_envs = 100
cfg.scene.env_spacing = 5.0

env = WaypointDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode="loaded",
                             checkpoint_path="logs/waypoint_loaded_v3/best_agent.pt")
agent.set_running_mode("eval")

num_envs = env.num_envs
episode_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))
max_eval_steps = episode_steps * 5

# Tracking
env_step = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_goals = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
prev_goal = env.goal_pos.clone()

results = {
    "goals_reached": [],
    "goals_per_second": [],
    "episode_length": [],
    "crashed": 0,
    "total_episodes": 0,
    "box_dropped": 0,
}

print(f"\n=== Stage 4 Loaded Eval (100 envs, physical box) ===\n")

obs, _ = env_wrapped.reset()

for step in range(max_eval_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_eval_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    ep_steps += 1

    goal = env.goal_pos
    goal_changed = (torch.abs(goal - prev_goal).sum(dim=-1) > 0.01)
    ep_goals += goal_changed.long()
    prev_goal = goal.clone()

    # Check box dropped
    if env.grasp_box is not None:
        pos_w = env.robot.data.root_pos_w
        box_pos = env.grasp_box.data.root_pos_w
        box_dist = torch.norm(pos_w - box_pos, dim=-1)

    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat

    if done_mask.any():
        for i in done_mask.nonzero(as_tuple=False).view(-1).tolist():
            results["total_episodes"] += 1
            n_goals = ep_goals[i].item()
            ep_len = ep_steps[i].item()
            results["goals_reached"].append(n_goals)
            results["episode_length"].append(ep_len)
            results["goals_per_second"].append(n_goals / max(ep_len / 150.0, 0.1))

            if term_flat[i] and pos_w[i, 2].item() < 0.20:
                results["crashed"] += 1

            if env.grasp_box is not None and box_dist[i].item() > 0.3:
                results["box_dropped"] += 1

            ep_goals[i] = 0
            ep_steps[i] = 0

    if step % 1000 == 0:
        print(f"  step {step:>5}  episodes={results['total_episodes']}")

# Results
n = max(results["total_episodes"], 1)

def _s(vals, fmt=".1f"):
    if not vals: return "n/a"
    return f"mean={sum(vals)/len(vals):{fmt}} med={sorted(vals)[len(vals)//2]:{fmt}} min={min(vals):{fmt}} max={max(vals):{fmt}}"

print(f"\n{'='*60}")
print(f"  STAGE 4 LOADED EVAL (physical box)")
print(f"{'='*60}")
print(f"  Total episodes:     {n}")
print(f"  Crashed:            {results['crashed']}/{n} ({100*results['crashed']/n:.0f}%)")
print(f"  Box dropped:        {results['box_dropped']}/{n} ({100*results['box_dropped']/n:.0f}%)")
print(f"")
print(f"  Goals reached/ep:   {_s(results['goals_reached'])}")
g = results['goals_reached']
print(f"    0 goals: {sum(1 for x in g if x==0)}/{n} ({100*sum(1 for x in g if x==0)/n:.0f}%)")
print(f"    1 goal:  {sum(1 for x in g if x==1)}/{n} ({100*sum(1 for x in g if x==1)/n:.0f}%)")
print(f"    2 goals: {sum(1 for x in g if x==2)}/{n} ({100*sum(1 for x in g if x==2)/n:.0f}%)")
print(f"    3+ goals:{sum(1 for x in g if x>=3)}/{n} ({100*sum(1 for x in g if x>=3)/n:.0f}%)")
print(f"  Goals/second:       {_s(results['goals_per_second'], '.2f')}")
print(f"  Episode length:     {_s(results['episode_length'], '.0f')} steps (max={episode_steps})")
print(f"{'='*60}")

env.close()
simulation_app.close()
