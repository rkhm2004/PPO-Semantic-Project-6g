# plot_perf_grouped_bars_paths.py
# Grouped bar charts of KPIs across Real/Hybrid/Sim2Real, one figure per agent.
# Reads metrics_computed.json from your explicit PATHS.
# Computes Collision Events (rate) from collision_events/(K*T) when needed.
# K (num_channels) and T (horizon) are inferred from multiple sources.

import os, json, re
import numpy as np
import matplotlib.pyplot as plt

# ====== YOUR EXACT FOLDERS (must contain metrics_computed.json) ======
PATHS = {
    "single": {
        "Real":     r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_single\20251101_111943",
        "Hybrid":   r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_single\fine_tuned_round2\20251101_221604",
        "Sim2Real": r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_single\eval_real_realistic\20251101_121048",
    },
    "multi": {
        "Real":     r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_marl\run_2025_09_27_20_51_22",
        "Hybrid":   r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_multi_agent\eval_real_unseen\20251102_003842",
        "Sim2Real": r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_multi_agent\fine_tuned\20251101_173244",
    },
}

# ====== OUTPUT ======
OUT_DIR = os.path.join("plots_of_cn", "performance_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# ====== KPI layout ======
# (Collision Events is included and computed if missing as a rate)
# logical_key, label, kind ("benefit" solid, "cost" hatched)
METRICS = [
    ("urllc_success",                 "URLLC Success",          "benefit"),
    ("mmtc_throughput",               "mMTC Throughput",        "benefit"),
    ("spectral_efficiency",           "Spectral Efficiency",    "benefit"),
    ("collision_rate_slots",          "Collision Slots (rate)", "cost"),
    ("collision_rate_channel_events", "Collision Events (rate)","cost"),
]

# Key aliases/fallbacks used if the primary name is missing in that JSON
KEY_ALIASES = {
    "urllc_success": [
        "urllc_success",
        "urllc_reliability",
        "urllc_score_scaled",
        "urllc_success_rate",
        "urllc_succ",
    ],
    "mmtc_throughput": [
        "mmtc_throughput", "mmtc_goodput", "mmtc_rate"
    ],
    "spectral_efficiency": [
        "spectral_efficiency", "spectral_eff", "se"
    ],
    "collision_rate_slots": [
        "collision_rate_slots", "collision_slots_rate", "slot_collision_rate"
    ],
    # For events-rate we may compute from raw collision_events/(K*T) if needed.
    "collision_rate_channel_events": [
        "collision_rate_channel_events", "collision_events_rate", "channel_collision_rate"
    ],
}

ENV_ORDER = ["Real", "Hybrid", "Sim2Real"]
ENV_COLOR = {"Real": (0.88,0.10,0.10), "Hybrid": (0.10,0.35,0.95), "Sim2Real": (0.05,0.05,0.05)}
ENV_LABEL = {"Real": "Train on Real", "Hybrid": "Sim2real & Finetuned(sim)", "Sim2Real": "Eval on Real"}

FIGSIZE = (11, 6.5)
BAR_WIDTH = 0.22
HATCH_COST = "//"
GRID_ALPHA = 0.25
VAL_FONTSIZE = 10

# ================= helpers =================
def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_metrics(run_dir):
    return read_json(os.path.join(run_dir, "metrics_computed.json"))

def read_env_cfg(run_dir):
    return read_json(os.path.join(run_dir, "env_config.json"))

_T_RE = re.compile(r"\bT=(\d+)\b")

def get_horizon_T(metrics_dict: dict, run_dir: str) -> float:
    """Infer T from direct keys, dataset_signature, or actions.npy shape."""
    if metrics_dict:
        for k in ("T", "horizon", "num_slots", "time_steps"):
            if k in metrics_dict:
                try:
                    return int(metrics_dict[k])
                except Exception:
                    pass
        sig = metrics_dict.get("dataset_signature", "")
        m = _T_RE.search(sig)
        if m:
            try: return int(m.group(1))
            except Exception: pass
    # actions.npy fallback
    anpy = os.path.join(run_dir, "actions.npy")
    if os.path.exists(anpy):
        try:
            arr = np.load(anpy, allow_pickle=False)
            return int(arr.shape[0])
        except Exception:
            pass
    return float("nan")

def read_env_num_channels(run_dir: str) -> int:
    """
    Find K (num_channels) in several ways:
    1) metrics_computed.json -> num_channels
    2) env_config.json -> num_channels
    3) infer from channel_split_active keys (exclude -1)
    4) fallback K=3
    """
    m = read_metrics(run_dir)
    if m:
        for k in ("num_channels", "channels", "K"):
            if k in m:
                try:
                    return int(m[k])
                except Exception:
                    pass
        csa = m.get("channel_split_active")
        if isinstance(csa, dict) and len(csa) > 0:
            try:
                keys = [int(k) for k in csa.keys() if int(k) >= 0]
                if keys:
                    return max(keys) + 1
            except Exception:
                pass
    cfg = read_env_cfg(run_dir)
    if cfg and "num_channels" in cfg:
        try: return int(cfg["num_channels"])
        except Exception: pass
    print(f"[WARN] num_channels not found in {run_dir}; assuming K=3")
    return 3

def pick_value(d, logical_key, num_channels=None, horizon_T=None):
    """
    Return (value, used_key) using aliases.
    Special-case: collision_rate_channel_events — if missing,
    compute collision_events/(K*T) or collision_events/T when K missing.
    """
    # Standard alias lookup
    for k in KEY_ALIASES.get(logical_key, [logical_key]):
        if d and (k in d):
            try:
                return float(d[k]), k
            except Exception:
                return np.nan, k

    # Special fallback computation for events rate
    if logical_key == "collision_rate_channel_events" and d:
        if "collision_events" in d:
            K = num_channels
            T = horizon_T
            if (K is not None) and (T is not None) and np.isfinite(K) and K > 0 and np.isfinite(T) and T > 0:
                return float(d["collision_events"]) / float(K * T), "collision_events/(K*T)"
            if (T is not None) and np.isfinite(T) and T > 0:
                return float(d["collision_events"]) / float(T), "collision_events/T (K missing)"
        return np.nan, None

    return np.nan, None

# ================= core =================
def collect_agent_values(agent):
    values = {}
    used   = {}
    for env in ENV_ORDER:
        folder = PATHS.get(agent, {}).get(env)
        if not folder:
            print(f"[WARN] {agent}/{env}: path not provided.")
            continue
        d = read_metrics(folder)
        if d is None:
            print(f"[WARN] Missing metrics_computed.json in {folder}")
            continue

        K = read_env_num_channels(folder)
        T = get_horizon_T(d, folder)

        vlist, klist = [], []
        for logical_key, _lbl, _kind in METRICS:
            v, used_key = pick_value(d, logical_key, num_channels=K, horizon_T=T)
            vlist.append(v); klist.append(used_key or "-")
        values[env] = vlist
        used[env]   = klist
    return values, used

def print_debug_table(agent, values, used):
    print(f"\n[DEBUG] Keys used for {agent}:")
    header = ["Metric"] + ENV_ORDER
    rowfmt = "{:<36} " + " | ".join(["{:<32}"]*len(ENV_ORDER))
    print(rowfmt.format(*header))
    print("-"* (36 + 3 + len(ENV_ORDER)*(32+3)))
    for idx, (_k, label, _kind) in enumerate(METRICS):
        row = [label]
        for env in ENV_ORDER:
            if env in used:
                uk = used[env][idx]
                val = values.get(env, [np.nan]*len(METRICS))[idx]
                if np.isfinite(val):
                    row.append(f"{uk} = {val:.4f}")
                else:
                    row.append(f"{uk or '—'} = NaN")
            else:
                row.append("missing env")
        print(rowfmt.format(*row))

def plot_grouped(agent, data):
    if not data:
        print(f"[SKIP] {agent}: no data to plot.")
        return

    n_metrics = len(METRICS)
    x = np.arange(n_metrics, dtype=float)

    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()

    handles = []; labels = []
    ymax = 1.0
    for i, env in enumerate(ENV_ORDER):
        if env not in data:
            continue
        vals = np.array(data[env], dtype=float)
        if np.isfinite(vals).any():
            ymax = max(ymax, float(np.nanmax(vals)))
        offs = x + (i - (len(ENV_ORDER)-1)/2) * BAR_WIDTH
        bars = ax.bar(offs, vals, width=BAR_WIDTH, color=ENV_COLOR[env], label=ENV_LABEL[env])
        # Hatch cost metrics
        for j, b in enumerate(bars):
            if METRICS[j][2] == "cost":
                b.set_hatch(HATCH_COST)
        # Value labels
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x()+b.get_width()/2, h + 0.015*max(1.0, ymax),
                        f"{h:.3f}", ha="center", va="bottom", fontsize=VAL_FONTSIZE)
        handles.append(bars); labels.append(ENV_LABEL[env])

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _k, lbl, _kind in METRICS], rotation=15, fontsize=16)
    ax.set_ylabel("Normalized Value", fontsize=16)
    ax.set_title(f"{agent.capitalize()} Agent — Performance Metrics", fontsize=16)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="y", labelsize=14) 
    ax.tick_params(axis="x", labelsize=14) 
    if handles:
        ax.legend([h[0] for h in handles], labels, loc="upper right", frameon=True, title="Environment")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{agent}_performance_bars.png")
    plt.savefig(out, dpi=300); plt.close()
    print(f"[OK] Saved {out}")

def main():
    for agent in ("single", "multi"):
        values, used = collect_agent_values(agent)
        print_debug_table(agent, values, used)
        plot_grouped(agent, values)

if __name__ == "__main__":
    main()
