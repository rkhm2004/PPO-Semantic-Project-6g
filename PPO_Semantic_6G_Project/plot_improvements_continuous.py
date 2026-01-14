# plot_improvements_continuous.py
# Continuous line of percentage improvements:
#  - First 4 points = our Hybrid vs Real improvements (URR, mMTC, SE) and
#    collision-slot reduction
#  - Remaining points = literature relative improvements from CSV

import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ---------- paths ----------
OURS_CSV = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Ours___absolute_KPIs.csv"
LIT_CSV  = r"D:\PROJECTS\SEM 3\CN\CN (2)\CN\PPO_Semantic_6G_Project\data\Literature___relative_improvements.csv"
OUT_DIR  = r"plots_of_cn"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG  = os.path.join(OUT_DIR, "line_continuous_improvements.png")

# ---------- column aliases ----------
ALIASES_OURS = {
    "setting": ["setting","env","environment","run","mode","config","method"],
    "urr": ["urr","urllc","urllc_success","urllc_reliability","urllc_success_rate","urllc_succ","urlcc"],
    "mmtc": ["mmtc","mmtc_throughput","mmtc_goodput","mmtc_rate"],
    "se": ["se","spectral_efficiency","spectral_eff","spectral efficiency"],
    "collisions": ["collision_slot_rate","collision_rate_slots","slot_collision_rate","collisions","collision_rate"]
}

# 👇 updated to match your CSV
ALIASES_LIT = {
    "paper": [
        "paper","work","source","ref","reference","title","citation",
        "citation_short","paper_id"  # added
    ],
    "metric": ["metric","kpi","name"],
    "value": [
        "relative_improvement_percent","relative_improvement_%","improvement_percent",
        "improvement_%","impr_%","rel_impr_%","value",
        "improvement_pct"  # added
    ]
}

def pick_col(df: pd.DataFrame, alias_list):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    # exact
    for a in alias_list:
        if a in low:
            return low[a]
    # contains
    for a in alias_list:
        for c in cols:
            if a in c.lower():
                return c
    return None

def load_ours(path):
    df = pd.read_csv(path)
    c_set = pick_col(df, ALIASES_OURS["setting"]) or "env"
    c_urr = pick_col(df, ALIASES_OURS["urr"]) or "urllc_success"
    c_mmt = pick_col(df, ALIASES_OURS["mmtc"]) or "mmtc_throughput"
    c_se  = pick_col(df, ALIASES_OURS["se"])  or "spectral_efficiency"
    c_col = pick_col(df, ALIASES_OURS["collisions"]) or "collision_slot_rate"

    s = df[c_set].astype(str).str.lower()
    ridx = s.str.contains("real") & ~s.str.contains("sim2real|sim-to-real|eval on real")
    hidx = s.str.contains("hybrid") | s.str.contains(r"sim\+real") | s.str.contains("fine") | s.str.contains("sim2real")

    real_row   = df.loc[ridx].head(1)
    hybrid_row = df.loc[hidx].head(1)
    if real_row.empty:
        real_row = df.loc[s.str.contains("real")].head(1)
    if hybrid_row.empty:
        hybrid_row = df.loc[s.str.contains("hybrid|sim2real|sim-to-real|fine|sim")].head(1)

    if real_row.empty or hybrid_row.empty:
        print("⚠️ Could not confidently pick Real/Hybrid rows — using first two rows as fallback.")
        if len(df) == 1:
            real_row = hybrid_row = df.iloc[[0]]
        else:
            real_row, hybrid_row = df.iloc[[0]], df.iloc[[1]]

    real = real_row[[c_urr,c_mmt,c_se,c_col]].iloc[0].astype(float)
    hybr = hybrid_row[[c_urr,c_mmt,c_se,c_col]].iloc[0].astype(float)

    return {
        "URR_impr_%": (hybr[c_urr]-real[c_urr]) / max(real[c_urr],1e-9) * 100.0,
        "mMTC_impr_%": (hybr[c_mmt]-real[c_mmt]) / max(real[c_mmt],1e-9) * 100.0,
        "SE_impr_%": (hybr[c_se]-real[c_se]) / max(real[c_se],1e-9) * 100.0,
        "Collisions_reduct_%": (real[c_col]-hybr[c_col]) / max(real[c_col],1e-9) * 100.0,
    }

def load_lit(path):
    d = pd.read_csv(path)
    p = pick_col(d, ALIASES_LIT["paper"])
    m = pick_col(d, ALIASES_LIT["metric"])
    v = pick_col(d, ALIASES_LIT["value"])
    if any(x is None for x in [p,m,v]):
        raise ValueError(
            "literature_relative.csv must include columns for paper/work, metric, and a percent value.\n"
            f"Found columns: {list(d.columns)}"
        )
    d = d.rename(columns={p:"paper", m:"metric", v:"relative_improvement_percent"})
    d["relative_improvement_percent"] = pd.to_numeric(d["relative_improvement_percent"], errors="coerce")
    d = d.dropna(subset=["relative_improvement_percent"])
    return d

# ----- compute
ours = load_ours(OURS_CSV)
lit  = load_lit(LIT_CSV)

labels = [
    "URR ↑ (URLLC Success) — Ours",
    "mMTC ↑ (Throughput) — Ours",
    "SE ↑ (Spectral Efficiency) — Ours",
    "Collisions ↓ (Slot Rate) — Ours",
]
values = [ours["URR_impr_%"], ours["mMTC_impr_%"], ours["SE_impr_%"], ours["Collisions_reduct_%"]]

for _, r in load_lit(LIT_CSV).iterrows():
    metric = str(r["metric"]).strip().lower()
    paper  = str(r["paper"]).strip()
    val    = float(r["relative_improvement_percent"])
    if metric.startswith("se"):
        lab = f"SE ↑ — {paper}"
    elif metric.startswith("mmtc"):
        lab = f"mMTC ↑ — {paper}"
    elif metric.startswith("urr"):
        lab = f"URR ↑ — {paper}"
    elif "collision" in metric or "slot" in metric:
        lab = f"Collisions ↓ — {paper}"
    else:
        lab = f"{r['metric']} — {paper}"
    labels.append(lab); values.append(val)

# ----- plot
plt.figure(figsize=(20, 8), dpi=150)
x = np.arange(len(values))
plt.plot(x, values, marker="o", linewidth=2.6, color="#d89000")
for i, v in enumerate(values):
    if np.isfinite(v):
        plt.text(i, v + (2 if v>=0 else -6), f"{v:.1f}", ha="center",
                 va="bottom" if v>=0 else "top", fontsize=9)
plt.xticks(x, labels, rotation=35, ha="right")
plt.ylabel("Improvement / Reduction (%)")
plt.title("Comparison of Improvements: Ours (Hybrid vs Real) followed by Literature", pad=12)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_PNG)
print(f"[OK] Saved {OUT_PNG}")
