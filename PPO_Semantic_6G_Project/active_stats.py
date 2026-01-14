import pandas as pd
df = pd.read_csv(r"data/real/real_traffic_holdout.csv")
if "slot" in df.columns:
    df = df.set_index("slot")
ue_cols = [c for c in df.columns if c.startswith("ue_")]
active = df[ue_cols].sum(axis=1)
print("UE columns:", ue_cols)
print("Slots:", len(df))
print("Active per slot -> min:", int(active.min()), "max:", int(active.max()), "mean:", round(active.mean(),2))
print("Histogram (active_count: num_slots):")
for k, v in active.value_counts().sort_index().items():
    print(f"  {int(k)}: {int(v)}")
