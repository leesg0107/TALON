"""Waypoint drone evaluation — measure reach accuracy and flight quality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import time
from envs.waypoint_cfg import WaypointEnvCfg
from envs.waypoint_env import WaypointDroneEnv
from controllers.drone_ctrl import quat_to_rot_matrix
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# ---- Config ----
MODE = "flight"  # "flight" or "loaded"
CKPT = "logs/waypoint_flight_v10/best_agent.pt"
NUM_ENVS = 100

cfg = WaypointEnvCfg(mode=MODE)
cfg.scene.num_envs = NUM_ENVS
# episode_length uses cfg default (20s) — NO override

env = WaypointDroneEnv(cfg=cfg)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode=MODE,
                             checkpoint_path=CKPT)
agent.set_running_mode("eval")

# ---- Tracking ----
num_envs = env.num_envs
max_eval_steps = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) * 5  # 5 episodes per env

# Per-env state
env_step = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_goals_reached = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
ep_goal_reach_steps = [[] for _ in range(num_envs)]  # steps when each goal was reached
prev_goal = env.goal_pos.clone()

# Results
results = {
    "goals_reached": [],          # goals reached per episode
    "goals_per_second": [],       # goals/sec throughput
    "time_to_first_goal": [],     # steps to reach first goal
    "time_between_goals": [],     # steps between consecutive goals
    "crashed": 0,
    "total_episodes": 0,
    "episode_length": [],
}

print(f"\n{'='*60}")
print(f"  Waypoint Drone Evaluation")
print(f"  Mode: {MODE}")
print(f"  Checkpoint: {CKPT}")
print(f"  Envs: {NUM_ENVS}, Episode: {cfg.episode_length_s}s")
print(f"{'='*60}\n")

obs, _ = env_wrapped.reset()

for step in range(max_eval_steps):
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_eval_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    env_step += 1
    ep_steps += 1

    # Metrics
    pos_w = env.robot.data.root_pos_w
    goal = env.goal_pos

    # Detect goal change (= goal reached, regenerated)
    # Use any() per-env: if ANY component changed, goal was regenerated
    goal_changed = (torch.abs(goal - prev_goal).sum(dim=-1) > 0.01)
    for i in goal_changed.nonzero(as_tuple=False).view(-1).tolist():
        ep_goals_reached[i] += 1
        ep_goal_reach_steps[i].append(ep_steps[i].item())
    prev_goal = goal.clone()

    # Episode end
    term_flat = terminated.view(-1)
    trunc_flat = truncated.view(-1)
    done_mask = term_flat | trunc_flat

    if done_mask.any():
        for i in done_mask.nonzero(as_tuple=False).view(-1).tolist():
            results["total_episodes"] += 1
            n_goals = ep_goals_reached[i].item()
            results["goals_reached"].append(n_goals)
            results["episode_length"].append(ep_steps[i].item())

            ep_len_sec = ep_steps[i].item() / 150.0
            results["goals_per_second"].append(n_goals / max(ep_len_sec, 0.1))

            # Time to first goal
            if ep_goal_reach_steps[i]:
                results["time_to_first_goal"].append(ep_goal_reach_steps[i][0] / 150.0)
                # Time between consecutive goals
                for j in range(1, len(ep_goal_reach_steps[i])):
                    dt = (ep_goal_reach_steps[i][j] - ep_goal_reach_steps[i][j-1]) / 150.0
                    results["time_between_goals"].append(dt)

            if term_flat[i] and pos_w[i, 2].item() < 0.20:
                results["crashed"] += 1

            # Reset per-env
            ep_goals_reached[i] = 0
            ep_steps[i] = 0
            ep_goal_reach_steps[i] = []

    if step % 500 == 0:
        cur_dist = torch.norm(pos_w - goal, dim=-1).mean().item()
        print(f"  step {step:>5}  dist={cur_dist:.3f}  episodes={results['total_episodes']}")

# ---- Results ----
import statistics
n = max(results["total_episodes"], 1)

def _s(vals, fmt=".3f"):
    if not vals: return "n/a"
    return f"mean={sum(vals)/len(vals):{fmt}} med={sorted(vals)[len(vals)//2]:{fmt}} min={min(vals):{fmt}} max={max(vals):{fmt}}"

print(f"\n{'='*60}")
print(f"  WAYPOINT EVAL RESULTS ({MODE})")
print(f"{'='*60}")
print(f"  Total episodes:       {n}")
print(f"  Crashed:              {results['crashed']}/{n} ({100*results['crashed']/n:.0f}%)")
print(f"")
print(f"  --- Goal Reaching ---")
print(f"  Goals reached/ep:     {_s(results['goals_reached'], '.1f')}")
g = results['goals_reached']
print(f"    0 goals: {sum(1 for x in g if x==0)}/{n} ({100*sum(1 for x in g if x==0)/n:.0f}%)")
print(f"    1 goal:  {sum(1 for x in g if x==1)}/{n} ({100*sum(1 for x in g if x==1)/n:.0f}%)")
print(f"    2 goals: {sum(1 for x in g if x==2)}/{n} ({100*sum(1 for x in g if x==2)/n:.0f}%)")
print(f"    3+ goals:{sum(1 for x in g if x>=3)}/{n} ({100*sum(1 for x in g if x>=3)/n:.0f}%)")
print(f"  Goals/second:         {_s(results['goals_per_second'], '.2f')}")
print(f"")
print(f"  --- Timing ---")
print(f"  Time to first goal:   {_s(results['time_to_first_goal'], '.2f')} sec")
print(f"  Time between goals:   {_s(results['time_between_goals'], '.2f')} sec")
print(f"  Episode length:       {_s(results['episode_length'], '.0f')} steps (max={int(cfg.episode_length_s/(cfg.sim.dt*cfg.decimation))})")
print(f"{'='*60}")

# Save
import json
from datetime import datetime

save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "thesis_data", "data_csv")
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"waypoint_eval_{MODE}_{timestamp}.json")
with open(save_path, 'w') as f:
    json.dump({
        "mode": MODE, "checkpoint": CKPT, "total_episodes": n,
        "crashed_pct": 100*results["crashed"]/n,
        "goals_reached": results["goals_reached"],
        "goals_per_second": results["goals_per_second"],
        "time_to_first_goal": results["time_to_first_goal"],
        "time_between_goals": results["time_between_goals"],
    }, f, indent=2)
print(f"\n  Saved: {save_path}")

env.close()
simulation_app.close()
