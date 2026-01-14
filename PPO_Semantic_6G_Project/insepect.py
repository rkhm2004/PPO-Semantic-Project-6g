import pandas as pd
import numpy as np

INIT = "data/real/real_initial_states.csv"
TRAF = "data/real/real_traffic_model.csv"
SEMS = "data/real/real_semantic_segments.csv"  # optional

def main():
    print("\n=== real_initial_states.csv ===")
    s = pd.read_csv(INIT)
    print(s.head())
    print("Columns:", list(s.columns))
    num_ues = s["ue_id"].nunique()
    print("Num UEs:", num_ues)
    if "traffic_type" in s.columns:
        print("Counts by traffic_type:\n", s["traffic_type"].value_counts(dropna=False))
    else:
        print("traffic_type not present -> env will default to first half URLLC, rest mMTC")

    print("\n=== real_traffic_model.csv ===")
    t = pd.read_csv(TRAF, index_col="slot")
    print(t.head())
    ue_cols = [c for c in t.columns if c.startswith("ue_")]
    print("UE columns:", ue_cols)
    print("Slots (T):", len(t))
    # basic 0/1 check
    unique_vals = np.unique(t.values)
    print("Unique values in traffic matrix:", unique_vals)
    active_per_slot = t[ue_cols].sum(axis=1)
    print("Active users per slot: min={}, max={}, mean={:.2f}".format(
        active_per_slot.min(), active_per_slot.max(), active_per_slot.mean()))
    act_rate = t[ue_cols].mean(axis=0)
    print("Per-UE activity rate (fraction of slots active):")
    for c, v in zip(ue_cols, act_rate):
        print(f"  {c}: {v:.3f}")

    print("\n=== real_semantic_segments.csv (optional) ===")
    try:
        sem = pd.read_csv(SEMS)
        print(sem.head())
        print("Columns:", list(sem.columns))
        print("Rows:", len(sem))
        if "importance_weight" in sem.columns:
            print("Importance stats:", sem["importance_weight"].describe())
    except FileNotFoundError:
        print("No semantic file found (ok).")
    except Exception as e:
        print("Could not parse semantic file:", e)

if __name__ == "__main__":
    main()
