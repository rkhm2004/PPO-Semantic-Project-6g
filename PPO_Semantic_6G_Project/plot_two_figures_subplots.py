# plot_two_figures_subplots.py
# Two figures, each with two subplots:
#   Fig A:  URR (URLLC success)  |  mMTC throughput
#   Fig B:  Spectral Efficiency  |  Collisions (shown as success = 1 - slot_rate)
#
# Inputs:
#   1) Ours___absolute_KPIs.csv  (uses ONLY the last row)
#        required columns:
#          method, env, urllc_success, mmtc_throughput, spectral_efficiency, collision_slot_rate
#   2) literature_relative.csv
#        required columns:
#          paper_id, citation_short, metric, improvement_pct, notes
#
# Notes:
# - Literature bars are relative improvements (percent -> fraction). Ours are absolute values.
# - All axes are normalized to [0, 1]; higher is better in every subplot.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= EDIT PATHS (keep the r-prefix for Windows) =========
OURS_CSV = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Ours___absolute_KPIs.csv"
LIT_CSV  = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Literature___relative_improvements.csv"
OUT_DIR  = os.path.join("plots_of_cn", "paper_compare")
os.makedirs(OUT_DIR, exist_ok=True)

# ========= Data loading helpers =========
def load_ours(path: str):
    df = pd.read_csv(path)
    needed = {"urllc_success", "mmtc_throughput", "spectral_efficiency", "collision_slot_rate"}
    miss = needed.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    last = df.tail(1).iloc[0]
    ours_label = str(last.get("env", "Ours (last)"))
    ours = {
        "urr": float(last["urllc_success"]),
        "mmtc": float(last["mmtc_throughput"]),
        "se": float(last["spectral_efficiency"]),
        # plot as success (higher better)
        "coll_success": 1.0 - float(last["collision_slot_rate"]),
    }
    return ours_label, ours

def load_lit(path: str):
    df = pd.read_csv(path)
    needed = {"paper_id", "citation_short", "metric", "improvement_pct"}
    miss = needed.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    # normalize metric labels
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    # relative improvement as fraction (e.g., 22.5 -> 0.225)
    df["value_rel"] = df["improvement_pct"].astype(float) / 100.0
    # if user wrote "collision reduction -25%", ensure positive "success" fraction
    is_coll = df["metric"].str.contains("coll")
    df.loc[is_coll, "value_rel"] = df.loc[is_coll, "value_rel"].abs()
    return df

# ========= Plotting helper =========
def plot_metric_subplot(ax, ours_label, ours_value, lit_rows, metric_pretty, palette):
    """
    ax: matplotlib axes to draw on
    ours_label: string for our bar
    ours_value: absolute value in [0,1]
    lit_rows: DataFrame rows for this metric; expects columns: citation_short, value_rel
    metric_pretty: x-axis label / title fragment
    palette: list of colors (first is ours)
    """
    names = [ours_label]
    vals  = [ours_value]
    colors = [palette[0]]

    if lit_rows is not None and len(lit_rows) > 0:
        for _, r in lit_rows.iterrows():
            names.append(str(r["citation_short"]))
            vals.append(float(r["value_rel"]))
            colors.append(palette[(len(colors)) % len(palette)])

    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, width=0.6)

    # value labels
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Normalized Value (higher is better)")
    ax.set_title(metric_pretty)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

# ========= Main =========
def main():
    ours_label, ours = load_ours(OURS_CSV)
    lit = load_lit(LIT_CSV)

    # mapping from metric tokens to pretty titles and lit filters
    # (we match literature rows by substring)
    metric_map = {
        "urr": {
            "pretty": "URLLC Success (URR)",
            "filter": lit["metric"].str.contains("urr|url", regex=True),
        },
        "mmtc": {
            "pretty": "mMTC Throughput",
            "filter": lit["metric"].str.contains("mmtc|through", regex=True),
        },
        "se": {
            "pretty": "Spectral Efficiency (SE or S-SE)",
            "filter": lit["metric"].str.contains(r"\bse\b|s-se|spectral", regex=True),
        },
        "coll_success": {
            "pretty": "Collision Success (1 − slot collision rate)",
            "filter": lit["metric"].str.contains("coll", regex=True),
        },
    }

    # Color palette (ours first)
    palette = [
        (0.12, 0.35, 0.95),   # ours
        (0.88, 0.10, 0.10),   # paper 1
        (0.10, 0.65, 0.35),   # paper 2
        (0.60, 0.25, 0.85),   # paper 3
        (0.95, 0.55, 0.15),   # paper 4
    ]

    # -------- Figure A: URR + mMTC --------
    figA, axesA = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    # URR
    rows_urr = lit.loc[metric_map["urr"]["filter"], ["citation_short", "value_rel"]]
    plot_metric_subplot(
        axesA[0],
        ours_label="Ours (URR)",
        ours_value=ours["urr"],
        lit_rows=rows_urr,
        metric_pretty=metric_map["urr"]["pretty"],
        palette=palette,
    )
    # mMTC
    rows_mmtc = lit.loc[metric_map["mmtc"]["filter"], ["citation_short", "value_rel"]]
    plot_metric_subplot(
        axesA[1],
        ours_label="Ours (mMTC)",
        ours_value=ours["mmtc"],
        lit_rows=rows_mmtc,
        metric_pretty=metric_map["mmtc"]["pretty"],
        palette=palette,
    )
    figA.suptitle("Figure A — URR and mMTC: Ours (absolute) vs Papers (relative)", fontsize=16, y=1.02)
    figA.savefig(os.path.join(OUT_DIR, "figure_A_URR_mMTC.png"), dpi=300)

    # -------- Figure B: SE + Collisions (success) --------
    figB, axesB = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    # SE
    rows_se = lit.loc[metric_map["se"]["filter"], ["citation_short", "value_rel"]]
    plot_metric_subplot(
        axesB[0],
        ours_label="Ours (SE)",
        ours_value=ours["se"],
        lit_rows=rows_se,
        metric_pretty=metric_map["se"]["pretty"],
        palette=palette,
    )
    # Collisions success
    rows_coll = lit.loc[metric_map["coll_success"]["filter"], ["citation_short", "value_rel"]]
    plot_metric_subplot(
        axesB[1],
        ours_label="Ours (Collisions↓ → Success↑)",
        ours_value=ours["coll_success"],
        lit_rows=rows_coll,
        metric_pretty=metric_map["coll_success"]["pretty"],
        palette=palette,
    )
    figB.suptitle("Figure B — SE and Collision Success: Ours (absolute) vs Papers (relative)", fontsize=16, y=1.02)
    figB.savefig(os.path.join(OUT_DIR, "figure_B_SE_Collisions.png"), dpi=300)

    print("[OK] Saved:")
    print(" -", os.path.join(OUT_DIR, "figure_A_URR_mMTC.png"))
    print(" -", os.path.join(OUT_DIR, "figure_B_SE_Collisions.png"))

if __name__ == "__main__":
    main()
