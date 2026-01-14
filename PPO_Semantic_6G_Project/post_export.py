# post_export.py
import os, json, argparse, numpy as np, pandas as pd
from App_marl import (find_latest_run, load_actions, load_traffic, load_initial_states,
                      reconstruct_assignments)

def jains_index(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    s = np.sum(x)
    s2 = np.sum(x**2)
    n = len(x)
    if n == 0 or s2 == 0:
        return 0.0
    return (s**2) / (n * s2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="outputs/output_marl")
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--traffic_csv", type=str, default="data/real/real_traffic_model.csv")
    ap.add_argument("--states_csv", type=str, default="data/real/real_initial_states.csv")
    ap.add_argument("--num_channels", type=int, default=3)
    ap.add_argument("--use_deconflict", type=int, default=1)
    args = ap.parse_args()

    run_dir = args.run if args.run else find_latest_run(args.runs_root)
    print("Using run:", run_dir)

    actions = load_actions(run_dir)
    traffic_df = load_traffic(args.traffic_csv)
    states_df = load_initial_states(args.states_csv)

    T = min(len(traffic_df), len(actions))
    actions = actions[:T, :]
    traffic_df = traffic_df.iloc[:T, :]

    assign = reconstruct_assignments(
        actions, traffic_df, states_df,
        num_channels=args.num_channels,
        use_deconflict=bool(args.use_deconflict)
    )

    # Save assignment matrix
    ue_ids = states_df["ue_id"].astype(int).tolist()
    df_assign = pd.DataFrame(assign, index=ue_ids, columns=[f"slot_{t}" for t in range(T)])
    df_assign.to_csv(os.path.join(run_dir, "assignments.csv"))

    # Per-slot stats
    slot_succ, slot_coll = [], []
    for t in range(T):
        ch = assign[:, t]
        active = (traffic_df.iloc[t].values[:assign.shape[0]] == 1)
        succ = 0
        coll = 0
        for ch_id in range(args.num_channels):
            users = np.where((ch == ch_id) & active)[0]
            if len(users) == 1:
                succ += 1
            elif len(users) > 1:
                coll += 1
        slot_succ.append(succ); slot_coll.append(coll)

    pd.DataFrame({
        "slot": np.arange(T),
        "successes": slot_succ,
        "collision_events": slot_coll
    }).to_csv(os.path.join(run_dir, "per_slot_stats.csv"), index=False)

    # Per-UE stats + fairness
    is_urllc = np.array([
        1 if states_df.loc[states_df.ue_id == i, "traffic_type"].values[0] == "URLLC" else 0
        for i in range(assign.shape[0])
    ], dtype=int)

    act = np.sum((traffic_df.values[:, :assign.shape[0]] == 1), axis=0).astype(int)
    served = np.zeros(assign.shape[0], dtype=int)

    for t in range(T):
        active = (traffic_df.iloc[t].values[:assign.shape[0]] == 1)
        ch = assign[:, t]
        for ch_id in range(args.num_channels):
            users = np.where((ch == ch_id) & active)[0]
            if len(users) == 1:
                served[users[0]] += 1

    rate = np.divide(served, np.maximum(act, 1), dtype=float)
    per_ue = pd.DataFrame({
        "ue_id": ue_ids,
        "traffic_type": ["URLLC" if is_urllc[i]==1 else "mMTC" for i in range(len(ue_ids))],
        "active": act,
        "served": served,
        "service_rate": rate
    })
    per_ue.to_csv(os.path.join(run_dir, "per_ue_stats.csv"), index=False)

    # Jain's fairness on mMTC only
    m_idx = np.where(is_urllc == 0)[0]
    fairness = jains_index(rate[m_idx]) if len(m_idx) > 0 else 0.0
    with open(os.path.join(run_dir, "summary_fairness.json"), "w") as f:
        json.dump({"jains_fairness_mmtc": float(fairness)}, f, indent=2)

    print("Exported CSVs + fairness to:", run_dir)

if __name__ == "__main__":
    main()
