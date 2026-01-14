import argparse, numpy as np, pandas as pd

def markov_onoff(T, p_on_to_off, p_off_to_on, seed=None, start_on=False):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, dtype=np.int8)
    state = 1 if start_on else 0
    for t in range(T):
        s[t] = state
        if state == 1:
            if rng.random() < p_on_to_off: state = 0
        else:
            if rng.random() < p_off_to_on: state = 1
    return s

def apply_quiet_windows(mat, rng, n_windows=2, w_min=10, w_max=25):
    T, N = mat.shape
    for _ in range(n_windows):
        start = rng.integers(low=0, high=max(1, T - w_max))
        length = int(rng.integers(w_min, w_max + 1))
        end = min(T, start + length)
        # reduce activity aggressively in the window
        for t in range(start, end):
            # keep at most 1 active (prefer URLLC if any active)
            active_idx = np.where(mat[t] == 1)[0].tolist()
            if len(active_idx) <= 1: continue
            # prefer to keep ue_0 or ue_1 if they are active
            keep = None
            for pref in [0, 1]:
                if pref in active_idx:
                    keep = pref; break
            if keep is None:
                keep = rng.choice(active_idx)
            for u in active_idx:
                mat[t, u] = 1 if u == keep else 0
    return mat

def apply_stress_spikes(mat, rng, n_spikes=2, s_min=2, s_max=3):
    T, N = mat.shape
    for _ in range(n_spikes):
        start = rng.integers(low=0, high=max(1, T - s_max))
        length = int(rng.integers(s_min, s_max + 1))
        end = min(T, start + length)
        # make many devices active (collisions likely)
        for t in range(start, end):
            # ensure URLLC on
            mat[t, 0] = 1; mat[t, 1] = 1
            # randomly activate ~half of mMTC
            for u in range(2, N):
                mat[t, u] = 1 if rng.random() < 0.6 else 0
    return mat

def soft_cap_active(mat, rng, target_mean=3.2):
    """Reduce mMTC activity to steer the average active UEs near target_mean (URLLC preserved)."""
    T, N = mat.shape
    act = mat.sum(axis=1).astype(float)
    current = act.mean()
    if current <= target_mean: 
        return mat  # already lean enough
    # thinning probability for mMTC rows (do not touch URLLC)
    # simple linear scaling
    over = min(0.9, (current - target_mean) / max(1.0, current))
    for t in range(T):
        if act[t] <= target_mean: 
            continue
        for u in range(2, N):
            if mat[t, u] == 1 and rng.random() < over:
                mat[t, u] = 0
    return mat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Existing real CSV (keeps slot count, columns)")
    ap.add_argument("--out_csv", required=True, help="Output path for realistic holdout")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ue_urllc", type=int, nargs=2, default=[0, 1], help="Indices for URLLC UEs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df_in = pd.read_csv(args.in_csv)
    if "slot" in df_in.columns:
        df_in = df_in.set_index("slot")
    ue_cols = [c for c in df_in.columns if c.startswith("ue_")]
    ue_cols = sorted(ue_cols, key=lambda x: int(x.split("_")[1]))
    N = len(ue_cols)
    T = len(df_in)
    if N < 6:
        # pad to 6, if needed
        for i in range(N, 6):
            df_in[f"ue_{i}"] = 0
        ue_cols = [f"ue_{i}" for i in range(6)]
        N = 6

    # Build new matrix with more realistic dynamics
    mat = np.zeros((T, N), dtype=np.int8)

    u0, u1 = args.ue_urllc
    # URLLC: frequent short bursts; start ON to simulate sustained demand
    mat[:, u0] = markov_onoff(T, p_on_to_off=0.08, p_off_to_on=0.35, seed=rng.integers(1e9), start_on=True)
    mat[:, u1] = markov_onoff(T, p_on_to_off=0.10, p_off_to_on=0.30, seed=rng.integers(1e9), start_on=True)

    # occasional "both urgent" windows (10–25 slots)
    for _ in range(3):
        start = rng.integers(0, max(1, T-25))
        end = min(T, start + int(rng.integers(10, 26)))
        mat[start:end, u0] = 1
        mat[start:end, u1] = 1

    # mMTC: long OFF, short ON, independent + rare group bursts
    for u in range(N):
        if u in (u0, u1): 
            continue
        # mostly OFF, sometimes ON
        mat[:, u] = markov_onoff(T, p_on_to_off=0.40, p_off_to_on=0.03, seed=rng.integers(1e9), start_on=False)

    # rare group mMTC bursts (sensors synchronized uploads)
    for _ in range(2):
        start = rng.integers(0, max(1, T-15))
        end = min(T, start + int(rng.integers(5, 16)))
        for u in range(2, N):
            if rng.random() < 0.6:  # not all sensors join
                mat[start:end, u] = 1

    # quiet windows (network relief)
    mat = apply_quiet_windows(mat, rng, n_windows=2, w_min=12, w_max=25)

    # stress spikes (all busy, collision-prone)
    mat = apply_stress_spikes(mat, rng, n_spikes=2, s_min=2, s_max=3)

    # softly reduce average load towards ~3.2 active UEs
    mat = soft_cap_active(mat, rng, target_mean=3.2)

    # Make sure we don't remove URLLC too aggressively
    # (Guarantee at least one URLLC is active in ~70% of slots)
    urllc_any = (mat[:, u0] + mat[:, u1]) > 0
    need = int(0.70*T) - int(urllc_any.sum())
    if need > 0:
        idx = np.where(~urllc_any)[0]
        rng.shuffle(idx)
        idx = idx[:need]
        mat[idx, u0] = 1  # turn on UE0 for these slots

    # Build output DataFrame
    out = pd.DataFrame(mat, columns=ue_cols, index=np.arange(T))
    out.index.name = "slot"
    out.to_csv(args.out_csv)

    # quick summary
    active_per_slot = out.sum(axis=1).values
    print(f"✅ Wrote: {args.out_csv}")
    print(f"Slots: {T}, UEs: {N}, mean active/slot: {active_per_slot.mean():.2f}")
    counts = np.bincount(active_per_slot.astype(int), minlength=N+1)
    print("Active-UE histogram (count of slots):")
    for k,v in enumerate(counts):
        if v: print(f"  {k} active: {v}")
