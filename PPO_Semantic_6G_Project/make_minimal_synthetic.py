# make_minimal_synthetic.py
# Build a realistic synthetic warm-up segment with active users per slot constrained to [2..5],
# then concatenate with the first N slots of your real traffic CSV.

import os
import argparse
import numpy as np
import pandas as pd


# --------------------------- helpers ---------------------------

def load_states(states_csv):
    df = pd.read_csv(states_csv)
    if "ue_id" not in df.columns:
        df["ue_id"] = np.arange(len(df))
    if "traffic_type" not in df.columns:
        # default: first half URLLC, rest mMTC
        n = len(df)
        df["traffic_type"] = ["URLLC" if i < n // 2 else "mMTC" for i in range(n)]
    return df


def load_real_traffic(csv):
    df = pd.read_csv(csv)
    if "slot" in df.columns:
        df = df.set_index("slot")
    ue_cols = [c for c in df.columns if c.startswith("ue_")]
    ue_cols = sorted(ue_cols, key=lambda x: int(x.split("_")[1]))
    return df[ue_cols].astype(int)


def empty_traffic(num_ues, T):
    cols = [f"ue_{i}" for i in range(num_ues)]
    return pd.DataFrame(0, index=np.arange(T), columns=cols, dtype=int)


def markov_onoff(T, p_on_to_off, p_off_to_on, seed=None, start_on=False):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=np.int8)
    state = 1 if start_on else 0
    for t in range(T):
        s[t] = state
        if state == 1:
            if rng.random() < p_on_to_off:
                state = 0
        else:
            if rng.random() < p_off_to_on:
                state = 1
    return s


def enforce_active_bounds(row, urllc_ids, mmtc_ids, min_active=2, max_active=5, rng=None):
    """Ensure number of actives in this slot is within [min_active, max_active].
       Preference: keep at least one URLLC active; trim/add mMTC first."""
    if rng is None:
        rng = np.random.default_rng()

    active_idx = np.where(row == 1)[0].tolist()
    k = len(active_idx)

    # Guarantee at least one URLLC active
    if all(row[u] == 0 for u in urllc_ids):
        # switch on one URLLC
        u_keep = int(rng.choice(urllc_ids))
        row[u_keep] = 1
        active_idx = np.where(row == 1)[0].tolist()
        k = len(active_idx)

    # If too few actives -> add mMTC first (then URLLC if needed)
    if k < min_active:
        pool = [u for u in mmtc_ids if row[u] == 0]
        rng.shuffle(pool)
        for u in pool:
            if k >= min_active:
                break
            row[u] = 1
            k += 1
        # If still below min and not enough mMTC, allow turning on another URLLC
        if k < min_active:
            pool_u = [u for u in urllc_ids if row[u] == 0]
            rng.shuffle(pool_u)
            for u in pool_u:
                if k >= min_active:
                    break
                row[u] = 1
                k += 1

    # If too many actives -> trim mMTC first, then (if required) trim URLLC but keep >=1 URLLC on
    if k > max_active:
        # turn off mMTC
        pool = [u for u in mmtc_ids if row[u] == 1]
        rng.shuffle(pool)
        for u in pool:
            if k <= max_active:
                break
            row[u] = 0
            k -= 1
        # if still too many, turn off URLLC except keep at least one
        if k > max_active:
            urllc_on = [u for u in urllc_ids if row[u] == 1]
            # keep one URLLC on
            if len(urllc_on) > 1:
                rng.shuffle(urllc_on)
                for u in urllc_on[1:]:
                    if k <= max_active or len([x for x in urllc_ids if row[x] == 1]) <= 1:
                        break
                    row[u] = 0
                    k -= 1

    return row


# ---------------------- synthetic builder ----------------------

def build_minimal_segment(states_df, T1=80, T2=120, seed=123,
                          min_active=2, max_active=5):
    """
    Phase A (T1): cleaner pattern (low collisions) with alternating URLLC.
    Phase B (T2): busier, overlaps & sparse mMTC; enforce active count in [min_active, max_active].
    """
    rng = np.random.default_rng(seed)
    num_ues = len(states_df)
    cols = [f"ue_{i}" for i in range(num_ues)]

    # Partition by type (fallback if not labeled)
    urllc_ids = [int(row.ue_id) for _, row in states_df.iterrows()
                 if str(row["traffic_type"]).upper().startswith("URLLC")]
    mmtc_ids = [int(row.ue_id) for _, row in states_df.iterrows()
                if not str(row["traffic_type"]).upper().startswith("URLLC")]
    if len(urllc_ids) == 0:
        urllc_ids = [0, 1] if num_ues >= 2 else [0]
        mmtc_ids = [i for i in range(num_ues) if i not in urllc_ids]

    # -------- Phase A: one URLLC at a time, with minimal mMTC
    dfA = empty_traffic(num_ues, T1)
    for t in range(T1):
        # alternate URLLC
        u = urllc_ids[t % len(urllc_ids)]
        dfA.at[t, f"ue_{u}"] = 1
        # occasionally add one mMTC to reach min_active bound
        row = dfA.iloc[t].values.copy()
        row = enforce_active_bounds(row, urllc_ids, mmtc_ids,
                                    min_active=min_active, max_active=max_active, rng=rng)
        dfA.iloc[t] = row

    # -------- Phase B: Markov-ish activity + bursts; then enforce bounds per slot
    dfB = empty_traffic(num_ues, T2)

    # URLLC base signals (start ON to emulate sustained demand)
    for idx, u in enumerate(urllc_ids):
        dfB[f"ue_{u}"] = markov_onoff(
            T2, p_on_to_off=0.12, p_off_to_on=0.30,
            seed=rng.integers(1e9), start_on=True if idx < 2 else False
        )

    # mMTC base signals (mostly off)
    for u in mmtc_ids:
        dfB[f"ue_{u}"] = markov_onoff(
            T2, p_on_to_off=0.45, p_off_to_on=0.05,
            seed=rng.integers(1e9), start_on=False
        )

    # Occasional URLLC both-active windows
    for _ in range(2):
        start = rng.integers(0, max(1, T2 - 25))
        end = min(T2, start + int(rng.integers(10, 26)))
        for u in urllc_ids[:2]:  # up to two URLLC prioritized
            dfB.iloc[start:end, dfB.columns.get_loc(f"ue_{u}")] = 1

    # Rare mMTC group bursts (not all join)
    for _ in range(2):
        start = rng.integers(0, max(1, T2 - 15))
        end = min(T2, start + int(rng.integers(5, 16)))
        for u in mmtc_ids:
            if rng.random() < 0.6:
                dfB.iloc[start:end, dfB.columns.get_loc(f"ue_{u}")] = 1

    # Enforce [min_active, max_active] per slot
    for t in range(T2):
        row = dfB.iloc[t].values.copy()
        row = enforce_active_bounds(row, urllc_ids, mmtc_ids,
                                    min_active=min_active, max_active=max_active, rng=rng)
        dfB.iloc[t] = row

    # Combine A + B
    df = pd.concat([dfA, dfB], axis=0)
    df.index.name = "slot"
    return df[cols].astype(int)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_csv", type=str, default="data/real/real_initial_states.csv")
    ap.add_argument("--traffic_real_csv", type=str, default="data/real/real_traffic_model.csv")
    ap.add_argument("--out_csv", type=str, default="data/controlled/controlled_traffic_model.csv")
    ap.add_argument("--T1", type=int, default=80, help="Phase A slots (clean/low-collision)")
    ap.add_argument("--T2", type=int, default=120, help="Phase B slots (busier)")
    ap.add_argument("--append_real", type=int, default=300, help="Append first N real slots (0=all, <0=none)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min_active", type=int, default=2)
    ap.add_argument("--max_active", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    states_df = load_states(args.states_csv)
    real_df   = load_real_traffic(args.traffic_real_csv)

    synth_df = build_minimal_segment(
        states_df,
        T1=args.T1, T2=args.T2,
        seed=args.seed,
        min_active=args.min_active,
        max_active=args.max_active
    )

    # Append part of real traffic (optional)
    final_df = synth_df
    if args.append_real >= 0:
        if args.append_real == 0 or args.append_real >= len(real_df):
            tail_real = real_df.copy()
        else:
            tail_real = real_df.iloc[:args.append_real].copy()
        tail_real.index = np.arange(len(synth_df), len(synth_df) + len(tail_real))
        # Ensure same columns and order (pad missing ues with 0)
        need_cols = [c for c in synth_df.columns]
        for c in need_cols:
            if c not in tail_real.columns:
                tail_real[c] = 0
        tail_real = tail_real[need_cols]
        final_df = pd.concat([synth_df, tail_real], axis=0)

    final_df.index.name = "slot"
    final_df.to_csv(args.out_csv)

    # Quick summary
    ue_cols = [c for c in final_df.columns if c.startswith("ue_")]
    active = final_df[ue_cols].sum(axis=1)
    print(f"✅ Augmented traffic saved to: {args.out_csv}")
    print(f"Total slots: {len(final_df)}, UEs: {len(ue_cols)}")
    print(f"Active/slot -> min: {int(active.min())}, max: {int(active.max())}, mean: {active.mean():.2f}")
    hist = active.value_counts().sort_index()
    print("Histogram (active_count: num_slots):")
    for k, v in hist.items():
        print(f"  {int(k)}: {int(v)}")


if __name__ == "__main__":
    main()
