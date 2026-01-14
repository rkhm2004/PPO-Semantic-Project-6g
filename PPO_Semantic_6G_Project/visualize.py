# visualize.py
# Reusable plotting helpers for both single-agent and multi-agent outputs.
# Updated for better color styling (pie/bar/heatmap) without changing logic.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

# Optional: clean figure background and fonts
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#aaaaaa",
    "grid.alpha": 0.3
})


def plot_channel_utilization(actions_or_assign, num_channels, output_path):
    """
    If given actions [T, num_ues], we flatten.
    If given assignments [num_ues, T] with -1 for inactive, we only count >=0.
    """
    arr = np.array(actions_or_assign)
    # Flatten either way
    used = arr.flatten()
    # Only count real channel indices
    used = used[used >= 0]

    # Compute counts per channel
    counts = [int(np.sum(used == c)) for c in range(num_channels)]
    labels = [f"Channel {c}" for c in range(num_channels)]

    # Use a bright, distinct palette
    colors = plt.cm.Set3(np.linspace(0, 1, num_channels))

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%0.1f%%", startangle=90, colors=colors,
        textprops={"color": "#222222"}
    )
    for auto in autotexts:
        auto.set_color("#111111")
        auto.set_fontweight("bold")

    ax.set_title("Channel Utilization", color="#222222")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_final_metrics(metrics: dict, output_path: str):
    labels = list(metrics.keys())
    values = [float(metrics[k]) for k in labels]

    # Bold colors for each metric
    colors = plt.cm.tab20(np.linspace(0, 1, len(values)))

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor="#222222", linewidth=0.6)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color="#222222", fontweight="bold")
    ax.set_ylim(0, max(1.05, max(values) + 0.1))
    ax.set_title("Metrics", color="#222222")
    ax.set_ylabel("Value", color="#222222")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20, color="#222222")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reward_components(assign, output_path):
    """
    Proxy reward visualization:
      +1 per slot per success (channel allocated to exactly one active UE),
      -1 per slot per colliding channel (>=2 active UEs on same channel)
    """
    T = assign.shape[1]
    pos, neg = [], []
    for t in range(T):
        used = assign[:, t]
        used = used[used >= 0]
        successes, collisions = 0, 0
        for ch in np.unique(used):
            n = np.sum(used == ch)
            if n == 1:
                successes += 1
            elif n > 1:
                collisions += 1
        pos.append(successes)
        neg.append(-collisions)

    pos_cum = np.cumsum(pos)
    neg_cum = np.cumsum(neg)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(T), pos_cum, label="Cumulative Success (+)",
            color="#1f77b4", linewidth=2.2)
    ax.plot(range(T), neg_cum, label="Cumulative Collisions (-)",
            color="#d62728", linewidth=2.2)
    ax.fill_between(range(T), pos_cum, alpha=0.08, color="#1f77b4")
    ax.fill_between(range(T), neg_cum, alpha=0.08, color="#d62728")

    ax.legend(loc="upper left", facecolor="white", edgecolor="#333333")
    ax.set_title("Reward Components (Proxy)", color="#222222")
    ax.set_xlabel("Time Slot", color="#222222")
    ax.set_ylabel("Cumulative Value", color="#222222")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_traffic_load(traffic_df: pd.DataFrame, output_path: str):
    active_per_slot = np.sum(traffic_df.values, axis=1)
    T = len(active_per_slot)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(T), active_per_slot, color="#2ca02c", linewidth=2.0)
    ax.set_title("Traffic Load Over Time", color="#222222")
    ax.set_xlabel("Time Slot", color="#222222")
    ax.set_ylabel("Active Users", color="#222222")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_channel_assignments(assign, states_df, output_path):
    """
    Heatmap of channel assignment per UE per slot.
      - Channel 0..2: colored (tab20)
      - Inactive (-1): gray
    """
    num_ues, T = assign.shape
    fig, ax = plt.subplots(figsize=(18, 7))

    # Build a custom ListedColormap: first 3 from tab20, last = gray for inactive
    ch_colors = list(plt.cm.tab20(np.linspace(0, 1, 3)))  # three channel colors
    inactive = (0.65, 0.65, 0.65, 1.0)                    # gray
    cmap = ListedColormap(ch_colors + [inactive])

    img = assign.copy()
    img[img < 0] = 3  # map inactive to color index 3
    im = ax.imshow(img, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=3)

    ax.set_xlabel("Time Slot", color="#222222")
    ax.set_ylabel("UE index", color="#222222")

    # Optional horizontal split: URLLC on top, mMTC below
    urllc_ids = [i for i in range(num_ues)
                 if states_df.loc[states_df.ue_id == i, "traffic_type"].values[0] == "URLLC"]
    if urllc_ids:
        mmtc_start = (max(urllc_ids) + 0.5)
        ax.axhline(y=mmtc_start, color='#aa0000', linestyle='--', linewidth=2)
        ax.text(T + 1, (mmtc_start / 2), "URLLC Users", va="center", color="#aa0000")
        ax.text(T + 1, (mmtc_start + (num_ues - mmtc_start) / 2), "mMTC Users", va="center", color="#aa0000")

    # Custom colorbar legend
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Channel 0", "Channel 1", "Channel 2", "Inactive"])
    cbar.outline.set_edgecolor("#333333")
    cbar.ax.tick_params(colors="#222222")

    ax.set_title("Channel Assignments by UE (Inactive = Gray)", color="#222222")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
