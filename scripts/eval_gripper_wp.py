"""Parallel headless eval for GripperWaypointEnv (Stage 1 flight & Stage 4 loaded).

Runs the SAME environment used for training — no transfer gap.
Measures goals reached per episode to verify model quality.

Usage:
  python scripts/eval_gripper_wp.py              # flight mode (Stage 1)
  python scripts/eval_gripper_wp.py --loaded     # loaded mode (Stage 4)
"""
import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--loaded", action="store_true", help="Stage 4 loaded mode")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=200, help="Total episodes to collect")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from envs.env_cfg import GripperDroneEnvCfg, Stage
from envs.gripper_waypoint_env import GripperWaypointEnv
from agents.waypoint_ppo_cfg import build_waypoint_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

MODE = "loaded" if args.loaded else "flight"

if MODE == "loaded":
    CKPT = "logs/gripper_wp_loaded_v21/best_agent.pt"
else:
    CKPT = "logs/gripper_wp_flight_v6/final_agent.pt"

NUM_ENVS = args.num_envs
TARGET_EPISODES = args.episodes

# ---- Environment (SAME as training) ----
cfg = GripperDroneEnvCfg(stage=Stage.GRASPING)
cfg.scene.num_envs = NUM_ENVS
cfg.scene.env_spacing = 5.0
cfg.episode_length_s = 20.0
cfg.lock_gripper = True
cfg.scene.grasp_object.spawn.rigid_props.kinematic_enabled = False
cfg.scene.grasp_object.spawn.rigid_props.disable_gravity = False

env = GripperWaypointEnv(cfg=cfg, mode=MODE)
env_wrapped = SkrlVecEnvWrapper(env)
device = env.device

agent = build_waypoint_agent(env=env_wrapped, device=device, mode=MODE,
                             checkpoint_path=CKPT)
agent.set_running_mode("eval")

# ---- Tracking ----
ep_goals = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
ep_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
prev_wp_idx = env.wp_idx.clone()

results = {
    "goals_reached": [],
    "episode_length_s": [],
    "crashed": 0,
    "total_episodes": 0,
}

print(f"\n{'='*60}")
print(f"  GripperWaypointEnv Eval — {MODE.upper()}")
print(f"  Checkpoint: {CKPT}")
print(f"  Envs: {NUM_ENVS}, Target episodes: {TARGET_EPISODES}")
print(f"  Episode: {cfg.episode_length_s}s")
print(f"{'='*60}\n")

obs, _ = env_wrapped.reset()
step = 0
max_steps = 500000  # safety limit

while results["total_episodes"] < TARGET_EPISODES and step < max_steps:
    with torch.no_grad():
        action = agent.act(obs, timestep=step, timesteps=max_steps)[0]

    obs, reward, terminated, truncated, info = env_wrapped.step(action)
    step += 1
    ep_steps += 1

    # Detect WP advances (wp_idx changed = goal reached)
    wp_advanced = (env.wp_idx != prev_wp_idx)
    ep_goals += wp_advanced.long()
    prev_wp_idx = env.wp_idx.clone()

    # Episode end
    done = terminated.view(-1) | truncated.view(-1)
    if done.any():
        for i in done.nonzero(as_tuple=False).view(-1).tolist():
            n_goals = ep_goals[i].item()
            ep_len_s = ep_steps[i].item() / 150.0
            results["goals_reached"].append(n_goals)
            results["episode_length_s"].append(ep_len_s)
            results["total_episodes"] += 1

            pos = env.robot.data.root_pos_w[i]
            local_z = (pos[2] - env.scene.env_origins[i, 2]).item()
            if terminated.view(-1)[i] and local_z < 0.20:
                results["crashed"] += 1

            ep_goals[i] = 0
            ep_steps[i] = 0

        if results["total_episodes"] >= TARGET_EPISODES:
            break

    if step % 3000 == 0:
        n = max(results["total_episodes"], 1)
        g = results["goals_reached"]
        avg = sum(g) / n if g else 0
        print(f"  step {step:>6} | episodes={n}/{TARGET_EPISODES} | "
              f"avg_goals={avg:.1f} | crashed={results['crashed']}")

# ---- Results ----
n = max(results["total_episodes"], 1)
g = results["goals_reached"]
avg_goals = sum(g) / n if g else 0
avg_len = sum(results["episode_length_s"]) / n if results["episode_length_s"] else 0
goals_per_sec = sum(g) / sum(results["episode_length_s"]) if results["episode_length_s"] else 0

print(f"\n{'='*60}")
print(f"  RESULTS — GripperWaypointEnv {MODE.upper()}")
print(f"{'='*60}")
print(f"  Total episodes:      {n}")
print(f"  Crashed:             {results['crashed']}/{n} ({100*results['crashed']/n:.0f}%)")
print(f"")
print(f"  --- Goals ---")
print(f"  Avg goals/episode:   {avg_goals:.1f}")
print(f"  Goals/second:        {goals_per_sec:.2f}")
print(f"  Avg episode length:  {avg_len:.1f}s")
print(f"")

# Distribution
print(f"  --- Distribution ---")
for k in range(max(g) + 1 if g else 0):
    cnt = sum(1 for x in g if x == k)
    if cnt > 0:
        print(f"    {k:>2} goals: {cnt:>4}/{n} ({100*cnt/n:5.1f}%)")

# Percentiles
if g:
    sg = sorted(g)
    p25 = sg[len(sg)//4]
    p50 = sg[len(sg)//2]
    p75 = sg[3*len(sg)//4]
    print(f"  p25={p25} p50={p50} p75={p75} min={min(g)} max={max(g)}")

print(f"{'='*60}")

env.close()
simulation_app.close()
