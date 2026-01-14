# resave_heatmaps_two.py
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

OUT_DIR = os.path.join("plots_of_cn", "channel_assignment")
os.makedirs(OUT_DIR, exist_ok=True)

def load_env_cfg(run_dir):
    cfg_path = os.path.join(run_dir, "env_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"env_config.json not found in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Backward-compatible defaults
    cfg.setdefault("num_channels", 3)
    return cfg

def load_traffic_df(path):
    df = pd.read_csv(path)
    if "slot" in df.columns:
        df = df.set_index("slot")
    return df

def active_mask_for_slot(df_row, num_ues):
    """
    Returns a boolean/int mask of length num_ues indicating which UEs are active.
    If ue_* columns are missing, assume all active (1).
    If fewer than num_ues ue_* columns exist, pad with zeros.
    """
    ue_cols = [c for c in df_row.index if isinstance(c, str) and c.startswith("ue_")]
    if len(ue_cols) == 0:
        return np.ones(num_ues, dtype=int)

    # Build mask from available columns (ordered by UE index)
    def parse_idx(c):
        try:
            return int(c.split("_", 1)[1])
        except Exception:
            return None
    pairs = [(parse_idx(c), int(df_row[c])) for c in ue_cols if parse_idx(c) is not None]
    # Initialize zeros then fill where we have data
    mask = np.zeros(num_ues, dtype=int)
    for idx, val in pairs:
        if 0 <= idx < num_ues:
            mask[idx] = 1 if val == 1 else 0
    return mask

def reconstruct_assignments(actions, traffic_df, num_channels):
    """
    actions: (T, num_ues) with channel id per UE (>=0) or possibly -1
    traffic_df: DataFrame of length >= T_use with ue_* columns (optional)
    Returns assign matrix of shape (num_ues, T_use) with {-1, 0..K-1}
    """
    T_actions, num_ues = actions.shape
    T_traffic = len(traffic_df)
    T_use = min(T_actions, T_traffic)
    if T_use == 0:
        raise ValueError("No overlap between actions and traffic (T_use=0).")

    assign = -1 * np.ones((num_ues, T_use), dtype=int)
    for t in range(T_use):
        active = active_mask_for_slot(traffic_df.iloc[t], num_ues)
        # cast action row to int, clip to valid range or keep as -1
        row = actions[t]
        # ensure row length matches num_ues
        if row.shape[0] != num_ues:
            # clamp/pad to match
            tmp = -1 * np.ones(num_ues, dtype=int)
            m = min(num_ues, row.shape[0])
            tmp[:m] = row[:m]
            row = tmp
        # fill assignments only for active UEs
        for u in range(num_ues):
            if active[u] == 1:
                assign[u, t] = int(row[u])
    return assign, T_use

def plot_heatmap(assign, num_channels, title, font, out_path):
    """
    assign: (num_ues, T_use) with {-1, 0..K-1}
    """
    # Fonts
    plt.rcParams.update({
        "xtick.labelsize": font,
        "ytick.labelsize": font,
        "axes.titlesize": font,
        "axes.labelsize": font
    })

    num_ues, T_use = assign.shape

    # Palette: one for Inactive (-1) + one per channel id
    base_colors = ["#ffeb3b", "#4B0082", "#31688e", "#35b779", "#a1c9f4", "#ff9f9b", "#8dd3c7", "#fb8072"]
    need = num_channels + 1
    if need > len(base_colors):
        base_colors = (base_colors * ((need // len(base_colors)) + 1))[:need]

    cmap = ListedColormap(base_colors[:need])
    boundaries = np.arange(-1.5, (num_channels - 0.5) + 1e-9, 1.0)  # -1,0,1,...,K-1 bins
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    plt.figure(figsize=(20, 7))
    im = plt.imshow(assign, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, origin="upper")

    plt.title(title)
    plt.xlabel("Time Slot")
    plt.ylabel("UE Index")

    cbar = plt.colorbar(im, ticks=np.arange(-1, num_channels, 1))
    cbar.ax.set_yticklabels(["Inactive"] + [f"Ch {i}" for i in range(num_channels)])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[OK] saved {out_path}")

def plot_one(run_dir, font=20, title="Channel Assignments"):
    # Load config & data
    cfg = load_env_cfg(run_dir)
    num_channels = int(cfg.get("num_channels", 3))
    traffic_csv = cfg["traffic_path"]
    traffic_df = load_traffic_df(traffic_csv)

    # actions.npy lives in the run folder
    acts_path = os.path.join(run_dir, "actions.npy")
    if not os.path.exists(acts_path):
        raise FileNotFoundError(f"actions.npy not found in {run_dir}")
    actions = np.load(acts_path)

    # Ensure actions is (T, num_ues)
    if actions.ndim != 2:
        raise ValueError(f"actions.npy must be 2D (T,num_ues); got shape {actions.shape}")

    # Reconstruct & clamp horizon
    assign, T_use = reconstruct_assignments(actions, traffic_df, num_channels)

    # Out name by caller
    return assign, num_channels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs=2, required=True,
                    help="Two run folders: first=single-agent run dir, second=multi-agent run dir")
    ap.add_argument("--font", type=int, default=22)
    ap.add_argument("--titles", nargs=2, default=["Single Agent — Real", "Multi Agent — Real"])
    args = ap.parse_args()

    # SINGLE
    assign1, K1 = plot_one(args.run_dirs[0], font=args.font, title=args.titles[0])
    out1 = os.path.join(OUT_DIR, "single_channel_assignments.png")
    plot_heatmap(assign1, K1, args.titles[0], args.font, out1)

    # MULTI
    assign2, K2 = plot_one(args.run_dirs[1], font=args.font, title=args.titles[1])
    out2 = os.path.join(OUT_DIR, "multi_channel_assignments.png")
    plot_heatmap(assign2, K2, args.titles[1], args.font, out2)

if __name__ == "__main__":
    main()
