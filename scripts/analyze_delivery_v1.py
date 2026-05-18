"""Offline analysis of logs/diagnose_delivery_raw.npz (no GPU needed).

Tests on the 8 features we already recorded:
  H2: Failure profile — spike (sudden action discontinuity) vs ramp (dynamics divergence)
  H6: Time-to-failure — how fast does tilt go 30° → 70°?
  H7: Lateral accel command distribution — what magnitude does the policy actually output?

Outputs:
  logs/diagnose_analysis/trajectory_comparison.png — sample success vs failure traces
  logs/diagnose_analysis/histograms.png — max_tilt, max_lat_accel distributions
  logs/diagnose_analysis/spike_vs_ramp_table.txt — categorization

Run:
    python scripts/analyze_delivery_v1.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "logs/diagnose_delivery_raw.npz"
OUT_DIR = "logs/diagnose_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Feature indices (diagnose_delivery.py):
TILT, LATA, VERTA, DRIFT, RELS, SSW, WPT, WPI = range(8)

# ----------------------------------------------------------------
# Load
# ----------------------------------------------------------------
d = np.load(DATA, allow_pickle=True)
traces = d['traces']
outcomes = d['outcomes']
records = [json.loads(r) for r in d['records']]

unique, counts = np.unique(outcomes, return_counts=True)
print(f"Loaded {len(traces)} delivery traces")
print(f"Outcomes: {dict(zip(unique, counts))}\n")

success_traces = [traces[i] for i, o in enumerate(outcomes) if o == 'full_success']
fail_traces = [(traces[i], outcomes[i]) for i, o in enumerate(outcomes) if 'delivery' in o]
print(f"Success: {len(success_traces)}, Delivery failures: {len(fail_traces)}\n")

# ----------------------------------------------------------------
# H2: Spike vs ramp classification
# ----------------------------------------------------------------
print("=" * 60)
print("  H2: Failure profile — spike vs ramp")
print("=" * 60)

SPIKE_JERK = 30.0   # m/s² change in one 6.67ms step
RAMP_RISE = 30.0    # m/s² accumulated rise
RAMP_WIN = 50       # frames (333ms)

spike, ramp, both, other = 0, 0, 0, 0
fail_categories = []
for tr, out in fail_traces:
    n = len(tr)
    if n < 10:
        other += 1; fail_categories.append("short"); continue
    end = tr[max(0, n - 60):]
    lat = end[:, LATA]
    jerk = np.abs(np.diff(lat))
    max_jerk = jerk.max() if len(jerk) else 0.0
    last_n = min(RAMP_WIN, len(lat))
    rise = lat[-1] - lat[-last_n] if last_n > 0 else 0.0

    is_spike = max_jerk > SPIKE_JERK
    is_ramp = rise > RAMP_RISE

    if is_spike and is_ramp:
        both += 1; cat = "both"
    elif is_spike:
        spike += 1; cat = "spike"
    elif is_ramp:
        ramp += 1; cat = "ramp"
    else:
        other += 1; cat = "other"
    fail_categories.append(cat)

total = len(fail_traces)
print(f"  Spike only (jerk > {SPIKE_JERK} m/s² in 1 step): {spike}/{total} ({100*spike/max(total,1):.1f}%)")
print(f"  Ramp only  (rise > {RAMP_RISE} m/s² over {RAMP_WIN} frames): {ramp}/{total} ({100*ramp/max(total,1):.1f}%)")
print(f"  Both spike + ramp:                                {both}/{total} ({100*both/max(total,1):.1f}%)")
print(f"  Neither (unclassified):                           {other}/{total} ({100*other/max(total,1):.1f}%)")

# Sub-breakdown by failure type
print("\n  Failure type × spike/ramp:")
ft_table = {}
for (tr, out), cat in zip(fail_traces, fail_categories):
    ft_table.setdefault(out, {"spike":0,"ramp":0,"both":0,"other":0,"short":0})
    ft_table[out][cat] += 1
for ft, cnts in sorted(ft_table.items(), key=lambda x: -sum(x[1].values())):
    tot = sum(cnts.values())
    print(f"    {ft:30s} total={tot:>3} | spike={cnts['spike']:>2} ramp={cnts['ramp']:>2} both={cnts['both']:>2} other={cnts['other']:>2}")

# ----------------------------------------------------------------
# H6: Time from 30° to 70° tilt (how recoverable was it?)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("  H6: Time-to-failure (30° → 70° tilt window)")
print("=" * 60)
t_window = []
for tr, _ in fail_traces:
    tilt = tr[:, TILT]
    o30 = np.where(tilt > 30)[0]
    o70 = np.where(tilt > 70)[0]
    if len(o30) and len(o70) and o70[0] > o30[0]:
        t_window.append(o70[0] - o30[0])
if t_window:
    t_window = np.array(t_window)
    print(f"  n_cases (reached 70°): {len(t_window)}")
    print(f"  frames: mean={t_window.mean():.0f}, med={np.median(t_window):.0f}, "
          f"p10={np.percentile(t_window, 10):.0f}, p90={np.percentile(t_window, 90):.0f}")
    print(f"  seconds @150Hz: mean={t_window.mean()/150:.3f}s, "
          f"med={np.median(t_window)/150:.3f}s, "
          f"p10={np.percentile(t_window,10)/150:.3f}s")
    print(f"\n  Interpretation: this is the recovery window. "
          f"If median < 0.1s, the catastrophe is too fast for reactive correction.")
else:
    print("  No traces reached 70°.")

# ----------------------------------------------------------------
# H7: Lateral accel command distribution
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("  H7: Lateral accel command magnitude")
print("=" * 60)
# What fraction of time does policy command > 1g, 3g, 5g?
all_lat_s = np.concatenate([tr[:, LATA] for tr in success_traces]) if success_traces else np.array([])
all_lat_f = np.concatenate([tr[:, LATA] for (tr, _) in fail_traces]) if fail_traces else np.array([])

def pct_above(arr, thr):
    if len(arr) == 0: return 0.0
    return 100.0 * (arr > thr).mean()

for thr, label in [(10, "1g"), (30, "3g"), (50, "5g"), (80, "8g")]:
    print(f"  Steps with |a_xy| > {thr} m/s² ({label}):")
    print(f"    success: {pct_above(all_lat_s, thr):5.2f}% of {len(all_lat_s)} steps")
    print(f"    fail:    {pct_above(all_lat_f, thr):5.2f}% of {len(all_lat_f)} steps")

print(f"\n  NOTE: action space is scaled to [-8, 8] m/s² nominally,")
print(f"        but Gaussian policy with unbounded space can output >> 1 (saturation).")
print(f"        Values > 8 m/s² come from policy commanding raw_action > 1.")

# ----------------------------------------------------------------
# Plots
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("  Generating plots…")
print("=" * 60)

# 1) Trajectory comparison: top-tilt fails vs sample success
fail_sorted = sorted(fail_traces, key=lambda x: -x[0][:, TILT].max())[:3]
fig, axes = plt.subplots(5, 2, figsize=(14, 14), sharex=False)
for i, (tr, out) in enumerate(fail_sorted):
    t = np.arange(len(tr)) / 150.0
    ax_t = axes[i, 0]
    ax_t.plot(t, tr[:, TILT], 'r-', linewidth=1.5)
    ax_t.axhline(26, color='gray', linestyle='--', alpha=0.5, label='r_tilt 26°')
    ax_t.axhline(70, color='black', linestyle='--', alpha=0.5, label='term 70°')
    ax_t.set_ylabel('tilt (deg)')
    ax_t.set_title(f'FAIL: {out} | max_tilt={tr[:, TILT].max():.0f}°')
    ax_t.legend(loc='upper left', fontsize=7)
    ax_t.grid(alpha=0.3)

    ax_a = axes[i, 1]
    ax_a.plot(t, tr[:, LATA], 'orange', linewidth=1.2, label='|lat_accel cmd|')
    ax_a.plot(t, tr[:, VERTA], 'b-', alpha=0.4, linewidth=1, label='vert_accel cmd')
    ax_a.axhline(30, color='gray', linestyle='--', alpha=0.5, label='3g')
    ax_a.set_ylabel('accel cmd (m/s²)')
    ax_a.legend(loc='upper left', fontsize=7)
    ax_a.grid(alpha=0.3)

# 2 sample successes
for i in range(2):
    if i >= len(success_traces): break
    tr = success_traces[i * (len(success_traces) // 3)]
    t = np.arange(len(tr)) / 150.0
    ax_t = axes[i + 3, 0]
    ax_t.plot(t, tr[:, TILT], 'g-', linewidth=1.5)
    ax_t.axhline(26, color='gray', linestyle='--', alpha=0.5)
    ax_t.axhline(70, color='black', linestyle='--', alpha=0.5)
    ax_t.set_ylabel('tilt (deg)')
    ax_t.set_title(f'SUCCESS sample #{i+1} | max_tilt={tr[:, TILT].max():.0f}°')
    ax_t.grid(alpha=0.3)

    ax_a = axes[i + 3, 1]
    ax_a.plot(t, tr[:, LATA], 'orange', linewidth=1.2)
    ax_a.plot(t, tr[:, VERTA], 'b-', alpha=0.4, linewidth=1)
    ax_a.axhline(30, color='gray', linestyle='--', alpha=0.5)
    ax_a.set_ylabel('accel cmd (m/s²)')
    ax_a.grid(alpha=0.3)
    if i == 1:
        ax_a.set_xlabel('time (s)')
        ax_t.set_xlabel('time (s)')

plt.tight_layout()
out_path = f"{OUT_DIR}/trajectory_comparison.png"
plt.savefig(out_path, dpi=100)
plt.close()
print(f"  Saved: {out_path}")

# 2) Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fail_max_tilt = [tr[:, TILT].max() for tr, _ in fail_traces]
succ_max_tilt = [tr[:, TILT].max() for tr in success_traces]
fail_max_lat = [tr[:, LATA].max() for tr, _ in fail_traces]
succ_max_lat = [tr[:, LATA].max() for tr in success_traces]

axes[0, 0].hist(succ_max_tilt, bins=30, alpha=0.5, label='success', color='blue', density=True)
axes[0, 0].hist(fail_max_tilt, bins=30, alpha=0.5, label='fail', color='red', density=True)
axes[0, 0].axvline(26, color='gray', linestyle='--', label='r_tilt 26°')
axes[0, 0].axvline(70, color='black', linestyle='--', label='term 70°')
axes[0, 0].set_xlabel('max_tilt during delivery (deg)')
axes[0, 0].set_ylabel('density')
axes[0, 0].set_title('Max Tilt distribution')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(succ_max_lat, bins=30, alpha=0.5, label='success', color='blue', density=True)
axes[0, 1].hist(fail_max_lat, bins=30, alpha=0.5, label='fail', color='red', density=True)
axes[0, 1].axvline(30, color='gray', linestyle='--', label='3g')
axes[0, 1].axvline(80, color='black', linestyle='--', label='8g (saturation)')
axes[0, 1].set_xlabel('max |lat_accel cmd| (m/s²)')
axes[0, 1].set_ylabel('density')
axes[0, 1].set_title('Max Lateral Accel Cmd distribution')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Time-to-failure window
if len(t_window):
    axes[1, 0].hist(t_window / 150.0, bins=30, color='red', alpha=0.7)
    axes[1, 0].axvline(np.median(t_window) / 150.0, color='black', linestyle='--',
                       label=f'median {np.median(t_window)/150:.2f}s')
    axes[1, 0].set_xlabel('time from tilt 30° to 70° (s)')
    axes[1, 0].set_ylabel('count')
    axes[1, 0].set_title('Recovery window: 30° → 70° transit time')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

# Jerk distribution: success vs fail final frames
def get_jerk(traces_, key=0):
    out_arr = []
    for tr in traces_:
        if isinstance(tr, tuple): tr = tr[0]
        if len(tr) < 5: continue
        out_arr.extend(np.abs(np.diff(tr[-60:, LATA])))
    return np.array(out_arr) if out_arr else np.array([0.0])
js = get_jerk(success_traces)
jf = get_jerk(fail_traces)
axes[1, 1].hist(js, bins=50, alpha=0.5, label='success (last 60 frames)', color='blue', density=True, range=(0, 60))
axes[1, 1].hist(jf, bins=50, alpha=0.5, label='fail (last 60 frames)', color='red', density=True, range=(0, 60))
axes[1, 1].axvline(SPIKE_JERK, color='black', linestyle='--', label=f'spike threshold {SPIKE_JERK}')
axes[1, 1].set_xlabel('|Δ lat_accel| per step (m/s²)')
axes[1, 1].set_ylabel('density')
axes[1, 1].set_title('Action jerk at episode end')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
out_path = f"{OUT_DIR}/histograms.png"
plt.savefig(out_path, dpi=100)
plt.close()
print(f"  Saved: {out_path}")

print("\nDone. Inspect the PNGs to see if failures are spikes or ramps.")
