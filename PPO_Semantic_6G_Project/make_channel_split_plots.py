# make_channel_split_plots_rescue.py
# Builds Sim-to-Real channel-split pies even if metrics/infos lack channel counts,
# by reconstructing assignments from actions.npy + traffic CSV.

import os, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- EDIT PATHS (same style you used before) ----------
PATHS = {
    "single": {
        "Sim2Real": r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_single\eval_only\20251012_121925",
    },
    "multi": {
        "Sim2Real": r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\outputs\output_multi_agent\eval_real\20251012_140630",
    }
}
# ------------------------------------------------------------

OUT_DIR = os.path.join("plots_of_cn", "performance_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

def read_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_metrics_counts(folder):
    """Try to read channel counts from metrics_computed.json (several possible keys)."""
    m = read_json(os.path.join(folder, "metrics_computed.json"))
    if not isinstance(m, dict):
        return {}
    for key in [
        "channel_split_active", "channel_split",
        "channel_counts", "channels_share", "channel_hist",
        "channel_time_share_percent", "channel_split_all"
    ]:
        if key in m and isinstance(m[key], dict):
            # values may be counts or percents; treat generically
            d = {}
            for k, v in m[key].items():
                try:
                    d[int(k)] = float(v)
                except Exception:
                    pass
            if d:
                return d
    return {}

def read_infos_counts(folder):
    """Try to build counts from infos.json (search for ch_0, ch_1, ... lists)."""
    i = read_json(os.path.join(folder, "infos.json"))
    if not isinstance(i, dict):
        return {}
    per_ue = {}
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    walk(v, kp)
                else:
                    if isinstance(v, list) and re.search(r"(?:^|[.\[])(?:ch_)(\d+)$", kp, re.I):
                        m = re.search(r"(?:^|[.\[])(?:ch_)(\d+)$", kp, re.I)
                        if m:
                            try:
                                per_ue[int(m.group(1))] = np.array(v, dtype=int)
                            except Exception:
                                pass
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                walk(v, f"{prefix}[{idx}]")
    walk(i)

    if not per_ue:
        return {}
    T = min(len(a) for a in per_ue.values())
    if T <= 0:
        return {}
    mat = np.stack([per_ue[u][:T] for u in sorted(per_ue.keys())], axis=0)  # U x T
    flat = mat.ravel()
    uniq, cnt = np.unique(flat, return_counts=True)
    return {int(k): float(c) for k, c in zip(uniq, cnt)}

def reconstruct_counts_from_actions(folder):
    """
    Last-resort: use actions.npy + env_config.json + traffic_csv to reconstruct
    UE×T assignments and count channel usage (active-only and inactive).
    """
    actions_path = os.path.join(folder, "actions.npy")
    env_cfg = read_json(os.path.join(folder, "env_config.json"))
    if not os.path.exists(actions_path) or not isinstance(env_cfg, dict):
        return {}

    num_channels = int(env_cfg.get("num_channels", 3))
    traffic_csv = env_cfg.get("traffic_path")
    if not traffic_csv or not os.path.exists(traffic_csv):
        return {}

    actions = np.load(actions_path)  # shape: T x U
    traffic_df = pd.read_csv(traffic_csv)
    if "slot" in traffic_df.columns:
        traffic_df = traffic_df.set_index("slot")

    T = len(traffic_df)
    U = actions.shape[1]
    if actions.shape[0] > T:
        actions = actions[:T, :]

    # Build UE x T assignment with -1 for inactive users
    assign = -1 * np.ones((U, actions.shape[0]), dtype=int)
    for t in range(actions.shape[0]):
        row = traffic_df.iloc[t]
        active = np.array([row.get(f"ue_{u}", 0) for u in range(U)], dtype=int)
        for u in range(U):
            if active[u] == 1:
                assign[u, t] = int(actions[t, u])

    # Count (include -1)
    flat = assign.ravel()
    uniq, cnt = np.unique(flat, return_counts=True)
    counts_all = {int(k): float(c) for k, c in zip(uniq, cnt)}
    # Prefer active-only if present
    active = flat[(flat >= 0) & (flat < num_channels)]
    if active.size:
        u2, c2 = np.unique(active, return_counts=True)
        counts_active = {int(k): float(c) for k, c in zip(u2, c2)}
        return counts_active
    return counts_all

def to_percentages(counts):
    s = sum(counts.values())
    if s <= 0:
        return {}
    return {k: 100.0 * (v / s) for k, v in counts.items()}

def plot_single_pie(agent, counts_pct, outname, include_inactive=False):
    if not counts_pct:
        print(f"[SKIP] {agent}: nothing to plot.")
        return
    labels = sorted(counts_pct)
    if include_inactive and -1 in labels:
        labels = [-1] + [l for l in labels if l != -1]
    else:
        labels = [l for l in labels if l >= 0]

    if not labels:
        print(f"[SKIP] {agent}: no active-channel labels.")
        return

    fracs = [counts_pct[l] for l in labels]
    total = sum(fracs) or 1.0
    fracs = [100.0*f/total for f in fracs]  # normalize again after filtering

    text_labels = ["Inactive" if l == -1 else f"Channel {l}" for l in labels]
    plt.figure(figsize=(7.6, 7.6))
    plt.title(f"{agent.capitalize()} — Channel Utilization (Sim-to-Real)")
    plt.pie(fracs, startangle=90,
            labels=text_labels,
            autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
            pctdistance=0.75, labeldistance=1.05)
    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.close()
    print(f"[OK] Saved {outname}")

def main():
    for agent in ("single", "multi"):
        folder = PATHS.get(agent, {}).get("Sim2Real")
        if not folder:
            print(f"[WARN] {agent}/Sim2Real: path not set.")
            continue

        # 1) Try metrics
        counts = read_metrics_counts(folder)
        if not counts:
            # 2) Try infos.json
            counts = read_infos_counts(folder)
        if not counts:
            # 3) Reconstruct from actions + traffic
            counts = reconstruct_counts_from_actions(folder)

        if not counts:
            print(f"[WARN] {agent}/Sim2Real: no channel counts found/derived in {folder}")
            continue

        # Prefer to show ONLY active channels in the pie (like your example)
        counts_pct = to_percentages(counts)
        out = os.path.join(OUT_DIR, f"{agent}_channel_pie_sim2real.png")
        plot_single_pie(agent, counts_pct, outname=out, include_inactive=False)

    print("\nDone. Check:", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
