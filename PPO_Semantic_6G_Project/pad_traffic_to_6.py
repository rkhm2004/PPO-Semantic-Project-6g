# save as pad_traffic_to_6.py and run once
import pandas as pd, sys
inp, outp = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp)
if "slot" in df.columns:
    df = df.set_index("slot")
have = [c for c in df.columns if c.startswith("ue_")]
need = [f"ue_{i}" for i in range(6)]
for c in need:
    if c not in df.columns:
        df[c] = 0
df = df[need]  # order
df.index.name = "slot"
df.to_csv(outp)
print("Wrote:", outp, "with columns:", list(df.columns))
