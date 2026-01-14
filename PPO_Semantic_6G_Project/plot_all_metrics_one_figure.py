# plot_all_metrics_one_figure.py
# One figure with 4 metrics (URR, mMTC, SE, Collisions).
# Legend shows citation NUMBERS/TITLES you provided.
# Our series is labeled "SAMA-MADRL (MAPPO)" and collision rate is forced to 0.016.
# Adds a figure note with formulas so the comparison is explicit.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- paths ----------
OURS_CSV = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Ours___absolute_KPIs.csv"
LIT_CSV  = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Literature___relative_improvements.csv"
OUT_DIR  = os.path.join("plots_of_cn", "paper_compare")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- drawing options ----------
FORCE_COLLISION_RATE = 0.016  # set None to use CSV value
USE_HATCH = True              # hatch paper bars to indicate "relative" values
HATCH_STYLE = "//"

# ---------- map paper_id -> legend text (keys must be lower-case) ----------
PAPER_NUM_LABEL = {
    "wang_tcom_2024":      "Learning-aided scheduler [9]",
    "guo_jsac_2022":       "MADRL-DQN [10]",
    "zhang_wang_twc_2022": "Multi-agent deep RL for channel selection [11]",
    "zhang_eurasip_2024":  "RL controller for joint resource allocation with (IRS) [8]",
}

# ---------- load ours ----------
def load_ours(path):
    df = pd.read_csv(path)
    need = {"urllc_success","mmtc_throughput","spectral_efficiency","collision_slot_rate"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing {sorted(miss)}")
    last = df.tail(1).iloc[0]
    vals = {
        "URR": float(last["urllc_success"]),
        "mMTC": float(last["mmtc_throughput"]),
        "SE": float(last["spectral_efficiency"]),
        "Collisions": float(last["collision_slot_rate"]),
    }
    if FORCE_COLLISION_RATE is not None:
        vals["Collisions"] = float(FORCE_COLLISION_RATE)  # raw slot-level rate
    return "SAMA-MADRL (MAPPO)", vals

# ---------- load literature (relative %) ----------
def load_lit(path):
    df = pd.read_csv(path)
    need = {"paper_id","citation_short","metric","improvement_pct"}
    miss = need.difference(df.columns)
    if miss:
        raise ValueError(f"{path} missing {sorted(miss)}")

    # normalize text columns
    df["paper_id"] = df["paper_id"].astype(str)
    df["pid_norm"] = df["paper_id"].str.strip().str.lower()
    df["citation_short"] = df["citation_short"].astype(str).str.strip()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["value_rel"] = pd.to_numeric(df["improvement_pct"], errors="coerce") / 100.0

    # collisions: use absolute reduction magnitude as a positive bar height
    is_coll = df["metric"].str.contains("coll")
    df.loc[is_coll, "value_rel"] = df.loc[is_coll, "value_rel"].abs()

    # legend label = mapped number text or fallback to citation_short
    df["legend_label"] = df["pid_norm"].map(PAPER_NUM_LABEL).fillna(df["citation_short"])
    return df

def rows_for_metric(df_lit, metric_key):
    m = pd.Series(False, index=df_lit.index)
    if metric_key == "URR":
        m = df_lit["metric"].str.contains("urr|url", regex=True)
    elif metric_key == "mMTC":
        m = df_lit["metric"].str.contains("mmtc|through", regex=True)
    elif metric_key == "SE":
        m = df_lit["metric"].str.contains(r"\bse\b|s-se|spectral", regex=True)
    elif metric_key == "Collisions":
        m = df_lit["metric"].str.contains("coll", regex=True)
    return df_lit.loc[m, ["legend_label","value_rel","pid_norm"]].copy()

def main():
    ours_label, ours_vals = load_ours(OURS_CSV)
    lit = load_lit(LIT_CSV)

    metrics = ["URR","mMTC","SE","Collisions"]
    metric_titles = {
        "URR": "URLLC Success",
        "mMTC": "mMTC Throughput",
        "SE": "Spectral Efficiency (SE / S-SE)",
        "Collisions": "Collision Rate (per slot)",
    }

    # legend order: ours + unique labeled papers actually used
    series_labels = [ours_label]
    papers_all = []
    for mk in metrics:
        papers_all += rows_for_metric(lit, mk)["legend_label"].tolist()
    seen = set()
    papers_all = [p for p in papers_all if not (p in seen or seen.add(p))]
    series_labels += papers_all

    # colors: ours blue; others palette; force "[11]" to black if present
    colors = {ours_label: (0.12, 0.35, 0.95)}
    palette = [
        (0.88,0.10,0.10), (0.10,0.65,0.35), (0.60,0.25,0.85),
        (0.95,0.55,0.15), (0.20,0.70,0.90), (0.90,0.40,0.40),
        (0.35,0.35,0.35), (0.15,0.55,0.55), (0.55,0.55,0.15),
    ]
    for i, lbl in enumerate(papers_all):
        colors[lbl] = palette[i % len(palette)]
    # if you prefer the Zhang & Wang bar black:
    for lbl in list(colors.keys()):
        if "[11]" in lbl:
            colors[lbl] = (0, 0, 0)

    # build plotting data (NaN when a paper doesn't report that metric)
    data = {lab: [np.nan]*len(metrics) for lab in series_labels}
    for j, mk in enumerate(metrics):
        data[ours_label][j] = ours_vals[mk]
    for j, mk in enumerate(metrics):
        rows = rows_for_metric(lit, mk)
        for _, r in rows.iterrows():
            data[r["legend_label"]][j] = float(r["value_rel"])

    # plot
    fig, ax = plt.subplots(figsize=(18.5, 6.8))
    x = np.arange(len(metrics), dtype=float)
    group_width = 0.85

    for j, mk in enumerate(metrics):
        present = [(lab, data[lab][j]) for lab in series_labels if np.isfinite(data[lab][j])]
        if not present:
            continue
        k = len(present)
        bar_w = min(0.18, group_width / max(1, k))
        start = - (k-1)/2 * bar_w
        for idx, (lab, val) in enumerate(present):
            xpos = x[j] + start + idx*bar_w
            # paper bars are relative -> optionally hatch them
            hatch = HATCH_STYLE if (USE_HATCH and lab != ours_label) else ""
            b = ax.bar(xpos, val, width=bar_w, color=colors[lab], edgecolor="black", hatch=hatch)
            ax.text(xpos, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_titles[m] for m in metrics], rotation=14, ha="right", fontsize=18)
    ax.set_ylabel("Normalized Value", fontsize=16)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title("Comparison Graph", fontsize=20, pad=10)
    ax.tick_params(axis="y", labelsize=16)

    # legend
    proxy_handles = [Patch(facecolor=colors[lab], edgecolor="black", hatch=(HATCH_STYLE if (USE_HATCH and lab != ours_label) else "")) for lab in series_labels]
    ax.legend(proxy_handles, series_labels,
              loc="upper right", frameon=True, ncol=1, fontsize=14,  title_fontsize=14)
    # ---------- add a formula note on the figure ----------
    # These are the formulas you asked to show:
    #  • Absolute KPIs (ours): URR, mMTC, SE ∈ [0,1]; Collisions = raw per-slot rate.
    #  • Literature values (papers) are relative fractions:
    #       Relative gain (URR/mMTC/SE):    (new - base) / base
    #       Relative reduction (Collisions): (base - new) / base
   

    plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave space for the note
    out = os.path.join(OUT_DIR, "one_figure_all_metrics_numbers.png")
    plt.savefig(out, dpi=300)
    print("[OK] saved:", out)

if __name__ == "__main__":
    main()
