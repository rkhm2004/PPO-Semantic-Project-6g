# setup_data.py
import pandas as pd
import numpy as np
import os
import json

# --- Raw input files (put them in ./data) ---
FILE_1 = 'data/6G_English_Education_Network_Traffic.csv'
FILE_2 = 'data/6G_English_Education_Traffic_20204.csv'

# --- Outputs ---
REAL_TIME_INITIAL_STATES   = 'data/real/real_initial_states.csv'
REAL_TIME_SEMANTIC_SEGMENTS= 'data/real/real_semantic_segments.csv'
REAL_TIME_TRAFFIC_MODEL    = 'data/real/real_traffic_model.csv'
CHANNEL_PARAMS_REAL        = 'data/channel_params_real.json'

def create_real_time_data(
    num_slots=300,
    top_n_ues=6,
    session_len_urllc=12,
    session_len_mmtc=18,
    min_active_per_slot=2,
    max_active_per_slot=3,
    seed=42
):
    """
    Build denser real traffic:
      - 6 UEs (2 URLLC + 4 mMTC)
      - Min active per slot = 2
      - Max active per slot = 3
      - URLLC sessions last 12 slots; mMTC sessions last 18 slots
    """
    rng = np.random.default_rng(seed)

    try:
        df1 = pd.read_csv(FILE_1)
        df2 = pd.read_csv(FILE_2)
    except Exception as e:
        raise FileNotFoundError(
            f"❌ Could not read raw files.\n{FILE_1}\n{FILE_2}\nError: {e}"
        )

    # Combine and filter
    combined = pd.concat([df1, df2], ignore_index=True)

    # Normalise columns
    if 'is_malicious' in combined.columns:
        combined = combined[combined['is_malicious'] == 0]
    # source id
    if 'source_ip' not in combined.columns:
        for c in ['src_ip', 'user_id', 'uid', 'src']:
            if c in combined.columns:
                combined['source_ip'] = combined[c].astype(str)
                break
    combined['source_ip'] = combined['source_ip'].astype(str)

    # timestamp
    ts_col = 'timestamp' if 'timestamp' in combined.columns else combined.columns[0]
    combined['timestamp'] = pd.to_datetime(combined[ts_col], errors='coerce')
    combined = combined.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # Pick top-N users
    top_users = combined['source_ip'].value_counts().nlargest(top_n_ues).index.tolist()
    if len(top_users) < top_n_ues:
        print(f"⚠️ Only found {len(top_users)} unique users, proceeding with those.")
        top_n_ues = len(top_users)
    selected = combined[combined['source_ip'].isin(top_users)].copy()
    id_map = {ip: i for i, ip in enumerate(top_users)}
    selected['ue_id'] = selected['source_ip'].map(id_map)

    # Initial states: first 2 URLLC, rest mMTC
    init_rows = []
    for ue in range(top_n_ues):
        ttype = 'URLLC' if ue < 2 else 'mMTC'
        prio  = 9 if ttype == 'URLLC' else 3
        init_rows.append({'ue_id': ue, 'traffic_type': ttype, 'priority': prio})
    init_df = pd.DataFrame(init_rows)
    os.makedirs('data/real', exist_ok=True)
    init_df.to_csv(REAL_TIME_INITIAL_STATES, index=False)

    # Build traffic model
    start, end = selected['timestamp'].min(), selected['timestamp'].max()
    total_seconds = max(1.0, (end - start).total_seconds())
    slot_dur = total_seconds / num_slots

    traf = pd.DataFrame(0, index=range(num_slots), columns=[f'ue_{i}' for i in range(top_n_ues)])
    # Mark sessions based on rows
    for _, row in selected.iterrows():
        ue = int(row['ue_id'])
        offset = (row['timestamp'] - start).total_seconds()
        slot = int(offset / slot_dur)
        if slot < 0 or slot >= num_slots:
            continue
        L = session_len_urllc if ue < 2 else session_len_mmtc
        end_slot = min(num_slots, slot + L)
        traf.loc[slot:end_slot-1, f'ue_{ue}'] = 1

    # --- Enforce min and max active per slot ---
    # Prefer to keep URLLC if we must drop
    mmtc_ids = list(range(2, top_n_ues))
    rr_idx = 0  # round-robin pointer for fill

    for s in range(num_slots):
        active_ids = [i for i in range(top_n_ues) if traf.at[s, f'ue_{i}'] == 1]
        # Cap to max_active_per_slot
        if len(active_ids) > max_active_per_slot:
            # Keep URLLC first
            urllc_kept = [i for i in active_ids if i < 2]
            rest = [i for i in active_ids if i >= 2]
            keep = urllc_kept[:]
            need_more = max_active_per_slot - len(urllc_kept)
            if need_more > 0:
                keep.extend(rest[:need_more])
            # Drop others
            for ue in active_ids:
                if ue not in keep:
                    traf.at[s, f'ue_{ue}'] = 0
        # Fill to min_active_per_slot
        elif len(active_ids) < min_active_per_slot:
            needed = min_active_per_slot - len(active_ids)
            for _ in range(needed):
                # add a mMTC user not yet active in this slot
                tried = 0
                while tried < len(mmtc_ids):
                    candidate = mmtc_ids[rr_idx % len(mmtc_ids)]
                    rr_idx += 1
                    tried += 1
                    if traf.at[s, f'ue_{candidate}'] == 0:
                        traf.at[s, f'ue_{candidate}'] = 1
                        active_ids.append(candidate)
                        break

            # Cap again in case we overflowed
            if len(active_ids) > max_active_per_slot:
                # Drop extras but keep URLLC
                active_ids = [i for i in range(top_n_ues) if traf.at[s, f'ue_{i}'] == 1]
                urllc_kept = [i for i in active_ids if i < 2]
                rest = [i for i in active_ids if i >= 2]
                keep = urllc_kept[:]
                need_more = max_active_per_slot - len(urllc_kept)
                if need_more > 0:
                    keep.extend(rest[:need_more])
                for ue in active_ids:
                    if ue not in keep:
                        traf.at[s, f'ue_{ue}'] = 0

    traf.index.name = 'slot'
    traf.to_csv(REAL_TIME_TRAFFIC_MODEL)

    # Semantic segments
    seg_rows = []
    if 'activity_label' in selected.columns:
        segmap = {'discussion': 0, 'stream': 1, 'quiz': 2, 'login': 3, 'VR-session': 4, 'submit_assignment': 5}
        for ue in range(top_n_ues):
            acts = selected.loc[selected['ue_id'] == ue, 'activity_label'].dropna().unique()
            for a in acts:
                if a in segmap:
                    seg_rows.append({'ue_id': ue, 'segment_id': segmap[a]})
    else:
        # fallback: random 2 segments per UE
        for ue in range(top_n_ues):
            picks = np.random.choice(range(6), size=2, replace=False)
            for p in picks:
                seg_rows.append({'ue_id': ue, 'segment_id': int(p)})
    seg_df = pd.DataFrame(seg_rows).drop_duplicates()
    seg_df.to_csv(REAL_TIME_SEMANTIC_SEGMENTS, index=False)

    print(f"✅ Saved {REAL_TIME_INITIAL_STATES}")
    print(f"✅ Saved {REAL_TIME_TRAFFIC_MODEL}")
    print(f"✅ Saved {REAL_TIME_SEMANTIC_SEGMENTS}")

def create_channel_params():
    params = {
        "bandwidth": 100e6, "carrier_frequency": 3.5e9, "num_channels": 3,
        "noise_spectral_density": -174, "max_transmit_power": 23,
        "path_loss_exponent": 3.5, "shadowing_std": 8,
        "urllc_requirements": {"max_latency_slots": 5, "reliability": 0.9999},
        "mmtc_requirements": {"min_throughput": 0.8, "target_throughput": 1.0}
    }
    with open(CHANNEL_PARAMS_REAL, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"✅ Saved {CHANNEL_PARAMS_REAL}")

if __name__ == "__main__":
    os.makedirs('data/real', exist_ok=True)
    create_real_time_data(
        num_slots=300,
        top_n_ues=6,
        session_len_urllc=12,
        session_len_mmtc=18,
        min_active_per_slot=2,
        max_active_per_slot=3,
        seed=42
    )
    create_channel_params()
    print("\n🎉 Real dataset (denser) generated successfully.")
