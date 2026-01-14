# plot_grouped_by_paper.py
# Grouped bar chart: x = 4 metrics, color = paper (Ours + literature).
# Uses last row from ours_kpis.csv. Converts collision rate to (1 - rate).
# For literature CSV:
#   (A) If absolute normalized columns are present: urr, mmtc, se, collisions_good (0..1; higher=better)
#   (B) Else if you only have rows with relative improvements (%): columns
#       paper_id/citation_short, metric, improvement_pct -> proxy normalized = improvement_pct/100

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== paths ====
OURS_CSV = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Ours___absolute_KPIs.csv"               # expects columns incl. urllc_success, mmtc_throughput, spectral_efficiency, collision_slot_rate
LIT_CSV  = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Literature___relative_improvements.csv"     # see formats A or B above
OUT_DIR  = r"plots_of_cn"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG  = os.path.join(OUT_DIR, "grouped_metrics_by_paper.png")

# ==== helpers ====
def pick(df, candidates, default=None):
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols: return cols[name]
    # partial match
    for name in candidates:
        for c in df.columns:
            if name in c.lower():
                return c
    return default

# ---- load ours (last row only) ----
def load_ours_lastrow(path):
    df = pd.read_csv(path)
    c_urr = pick(df, ["urllc_success","urllc_reliability","urllc_success_rate","urr"])
    c_mmt = pick(df, ["mmtc_throughput","mmtc_goodput","mmtc_rate","mmtc"])
    c_se  = pick(df, ["spectral_efficiency","se"])
    c_col = pick(df, ["collision_slot_rate","collision_rate_slots","slot_collision_rate","collisions"])
    if None in (c_urr, c_mmt, c_se, c_col):
        raise ValueError(f"Missing required columns in ours_kpis.csv; found: {list(df.columns)}")
    row = df.iloc[-1]
    urr = float(row[c_urr])
    mmtc = float(row[c_mmt])
    se   = float(row[c_se])
    # convert cost metric to "higher is better"
    col_good = 1.0 - float(row[c_col])
    return {"paper":"Ours (Hybrid on Real)", "URR":urr, "mMTC":mmtc, "SE":se, "Collisions↓":col_good}

# ---- load literature (absolute or relative) ----
def load_literature(path):
    d = pd.read_csv(path)

    # Absolute mode?
    have_abs = all(x in [c.lower() for c in d.columns]
                   for x in ["urr","mmtc","se","collisions_good"])
    if have_abs:
        c_paper = pick(d, ["citation_short","paper","paper_id","source","work","title"], default=d.columns[0])
        return [
            {"paper": str(r[c_paper]),
             "URR":  float(r[pick(d,["urr"])]),
             "mMTC": float(r[pick(d,["mmtc"])]),
             "SE":   float(r[pick(d,["se"])]),
             "Collisions↓": float(r[pick(d,["collisions_good"])])
            }
            for _, r in d.iterrows()
        ]

    # Relative mode (per-row metric & improvement %)
    c_paper = pick(d, ["citation_short","paper","paper_id","source","work","title"], default=None)
    c_metric = pick(d, ["metric","kpi","name"], default=None)
    c_imp = pick(d, ["improvement_pct","improvement_percent","relative_improvement_percent","value"], default=None)
    if None in (c_paper, c_metric, c_imp):
        raise ValueError("literature_relative.csv must have either absolute columns (urr,mmtc,se,collisions_good) "
                         "or per-row metric with improvement_pct + paper.")

    d = d.copy()
    d["__paper"] = d[c_paper].astype(str)
    d["__metric"] = d[c_metric].astype(str).str.lower()
    d["__val"] = pd.to_numeric(d[c_imp], errors="coerce") / 100.0  # proxy normalized in 0..1

    # reduce to {paper: {metric_name -> value}}
    grouped = {}
    for _, r in d.iterrows():
        paper = r["__paper"]
        m = r["__metric"]
        v = r["__val"]
        if pd.isna(v): continue
        key = None
        if m.startswith("se"): key = "SE"
        elif m.startswith("mmtc"): key = "mMTC"
        elif m.startswith("urr") or "urllc" in m: key = "URR"
        elif "collision" in m or "slot" in m: key = "Collisions↓"
        if key is None: continue
        grouped.setdefault(paper, {})[key] = float(v)

    # Convert to list of dicts, missing metrics -> NaN
    rows = []
    metrics = ["URR","mMTC","SE","Collisions↓"]
    for paper, mp in grouped.items():
        rows.append({"paper":paper, **{k: mp.get(k, np.nan) for k in metrics}})
    return rows

# ---- assemble data frame for plotting ----
ours_row = load_ours_lastrow(OURS_CSV)
lit_rows = load_literature(LIT_CSV)

df = pd.DataFrame([ours_row] + lit_rows)
# Order columns
metrics = ["URR","mMTC","SE","Collisions↓"]
papers  = df["paper"].tolist()

# ---- plot grouped bars ----
plt.figure(figsize=(14, 7), dpi=150)
x = np.arange(len(metrics))
bar_w = 0.8 / len(papers)  # keep total width ~0.8

# choose a palette
PALETTE = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
    "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac"
]
while len(PALETTE) < len(papers):
    PALETTE = PALETTE + PALETTE  # extend if many papers

handles = []
for i, paper in enumerate(papers):
    vals = df.loc[i, metrics].to_numpy(dtype=float)
    offs = x - 0.4 + (i + 0.5) * bar_w
    bars = plt.bar(offs, vals, width=bar_w, label=paper, color=PALETTE[i])
    # annotate
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            plt.text(b.get_x()+b.get_width()/2, h + 0.02, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=8)

plt.xticks(x, metrics, fontsize=11)
plt.yticks(fontsize=11)
plt.ylabel("Normalized Value (higher is better)", fontsize=12)
plt.title("Performance Metrics — Ours (last row) vs Papers", fontsize=14)
plt.grid(axis="y", alpha=0.25)
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, title="Method / Paper", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"[OK] Saved {OUT_PNG}")
