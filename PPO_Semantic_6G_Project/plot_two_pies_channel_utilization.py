import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.path.join("plots_of_cn", "channel_utilization")
os.makedirs(OUT_DIR, exist_ok=True)

def find_col(df, wanted_names):
    norm = {c: str(c).strip().lower().replace("  ", " ").replace("\t", " ") for c in df.columns}
    for want in wanted_names:
        want_norm = want.strip().lower()
        for c, c_norm in norm.items():
            if c_norm == want_norm:
                return c
    for want in wanted_names:
        want_norm = want.strip().lower()
        for c, c_norm in norm.items():
            if want_norm in c_norm:
                return c
    return None

def parse_percent_col(s):
    ser = s.astype(str).str.strip().str.replace("%", "", regex=False)
    ser = pd.to_numeric(ser, errors="coerce")
    if ser.max() is not np.nan and ser.max() is not None and ser.max() <= 1.0:
        ser = ser * 100.0
    return ser.fillna(0.0)

def plot_pie(values, labels, title, outfile):
    total = float(np.sum(values))
    if total <= 0:
        print(f"[WARN] {title}: all zeros; skip.")
        return

    pct = np.array(values, dtype=float)
    pct = pct / pct.sum() * 100.0

    # Force Channel 2 to bottom center (270°)
    try:
        idx_channel_2 = labels.index("2")
        angle_offset = 270 - (pct[:idx_channel_2].sum() + pct[idx_channel_2] / 2) * 360 / 100
    except ValueError:
        angle_offset = 90

    colors = plt.cm.tab20.colors[:len(labels)]
    explode = [0.02] * len(labels)

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,  # Reduced title size
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    wedges, texts, autotexts = ax.pie(
        pct,
        labels=[f"Channel {l}" for l in labels],
        autopct="%.1f%%",
        startangle=angle_offset,
        counterclock=False,
        pctdistance=0.5,
        labeldistance=1.15,
        textprops={"fontsize": 15},
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("bold")

    for text in texts:
        text.set_fontweight("bold")  # Make channel names bold
        text.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

    ax.set_title(title, pad=14)
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to CSV file with channel utilization data")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    ch_col = find_col(df, ["channel"])
    sa_col = find_col(df, ["singel agent", "single agent", "single"])
    ma_col = find_col(df, ["multi agent", "multi"])

    if ch_col is None or sa_col is None or ma_col is None:
        raise ValueError(
            f"Could not detect expected columns.\n"
            f"Found: {list(df.columns)}\n"
            f"Need something like: 'channel', 'Singel Agent' (or 'Single Agent'), 'Multi Agent'."
        )

    channels = df[ch_col].astype(str).str.strip().tolist()
    single_vals = parse_percent_col(df[sa_col])
    multi_vals  = parse_percent_col(df[ma_col])

    plot_pie(
        values=single_vals.values,
        labels=channels,
        title="",
        outfile=os.path.join(OUT_DIR, "channel_utilization_single.png"),
    )

    plot_pie(
        values=multi_vals.values,
        labels=channels,
        title="",
        outfile=os.path.join(OUT_DIR, "channel_utilization_multi.png"),
    )

if __name__ == "__main__":
    main()