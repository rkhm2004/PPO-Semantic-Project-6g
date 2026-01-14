import pandas as pd
import numpy as np
import os
import datetime
from collections import defaultdict

# --- Define file paths ---
file1 = 'data/6G_English_Education_Network_Traffic.csv'
file2 = 'data/6G_English_Education_Traffic_20204.csv'
initial_states_output = 'data/initial_states.csv'
traffic_model_output = 'data/traffic_model.csv'
semantic_segments_output = 'data/semantic_segments.csv'
channel_params_output = 'data/channel_params.json'

# --- Step 1: Load and combine the datasets ---
print("--- Step 1: Loading and combining datasets ---")
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Clean the data by removing malicious traffic
combined_df = combined_df[combined_df['is_malicious'] == 0]
combined_df['source_ip'] = combined_df['source_ip'].astype(str)
combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], format='mixed')
combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)

# --- Step 2: GUARANTEED selection of a balanced subset of users ---
print("\n--- Step 2: Selecting a guaranteed, balanced subset ---")
# These IPs are known to have URLLC-like activities in the dataset.
urllc_ips = ['192.168.2.220', '192.168.2.149', '192.168.1.189']
# These IPs are chosen to have only mMTC-like activities.
mmtc_ips = ['192.168.1.172', '192.168.1.113', '192.168.1.139']

selected_ips = urllc_ips + mmtc_ips
selected_df = combined_df[combined_df['source_ip'].isin(selected_ips)].copy()

# Map the selected IPs to new, sequential UE IDs (0 to 5)
new_id_map = {old_ip: new_id for new_id, old_ip in enumerate(selected_ips)}
selected_df['ue_id'] = selected_df['source_ip'].map(new_id_map)
selected_df = selected_df.sort_values(by='ue_id').reset_index(drop=True)

print(f"Selected {len(urllc_ips)} URLLC users and {len(mmtc_ips)} mMTC users.")

# --- Step 3: Create initial_states.csv ---
print("\n--- Step 3: Creating initial_states.csv ---")
initial_states_list = []
total_ues = len(selected_ips)

for ue_id in range(total_ues):
    ip = selected_ips[ue_id]
    traffic_type = 'URLLC' if ip in urllc_ips else 'mMTC'
    priority = 9 if traffic_type == 'URLLC' else 3
    
    initial_states_list.append({
        'ue_id': ue_id,
        'ue_type': 'device',
        'x_pos': np.random.randint(0, 500),
        'y_pos': np.random.randint(0, 500),
        'traffic_type': traffic_type,
        'priority': priority,
        'data_size': np.random.randint(32, 256)
    })
    
initial_states_df = pd.DataFrame(initial_states_list)
initial_states_df.to_csv(initial_states_output, index=False)
print(f"initial_states.csv has been created at: {initial_states_output}")
print(initial_states_df)

# --- Step 4: Create traffic_model.csv ---
print("\n--- Step 4: Creating traffic_model.csv ---")
num_slots = 1000
slot_duration = (selected_df['timestamp'].max() - selected_df['timestamp'].min()).total_seconds() / num_slots
start_time = selected_df['timestamp'].min()
    
traffic_model_df = pd.DataFrame(index=range(num_slots))
    
for ue_id in range(total_ues):
    ue_data = selected_df[selected_df['ue_id'] == ue_id]
    is_active = np.zeros(num_slots, dtype=int)
    
    for _, row in ue_data.iterrows():
        time_diff = (row['timestamp'] - start_time).total_seconds()
        slot = int(time_diff // slot_duration)
        if 0 <= slot < num_slots:
            is_active[slot] = 1
    
    traffic_model_df[f'ue_{ue_id}'] = is_active
    
traffic_model_df.index.name = 'slot'
traffic_model_df.to_csv(traffic_model_output)
print(f"traffic_model.csv has been created at: {traffic_model_output}")
print(traffic_model_df.head())

# --- Step 5: Create semantic_segments.csv ---
print("\n--- Step 5: Creating semantic_segments.csv ---")
activity_to_segment = {
    'discussion': 0, 'stream': 1, 'quiz': 2, 'login': 3,
    'VR-session': 4, 'submit_assignment': 5
}
    
semantic_data_list = []
    
for ue_id in selected_df['ue_id'].unique():
    ue_activities = selected_df[selected_df['ue_id'] == ue_id]['activity_label'].unique()
    
    for activity in ue_activities:
        if activity in activity_to_segment:
            semantic_data_list.append({
                'ue_id': ue_id,
                'segment_id': activity_to_segment[activity]
            })

semantic_df = pd.DataFrame(semantic_data_list)
semantic_df.to_csv(semantic_segments_output, index=False)
print(f"semantic_segments.csv has been created at: {semantic_segments_output}")
print(semantic_df)