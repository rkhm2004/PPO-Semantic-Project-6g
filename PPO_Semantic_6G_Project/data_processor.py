import pandas as pd
import numpy as np
import os
import datetime
from collections import defaultdict

def create_controlled_data():
    """Creates the controlled, static configuration files."""
    # Define the static data for the controlled environment
    initial_states_output = 'data/initial_states.csv'
    semantic_segments_output = 'data/semantic_segments.csv'
    traffic_model_output = 'data/traffic_model.csv'
    
    # initial_states.csv
    initial_states_df = pd.DataFrame({
        'ue_id': [0, 1, 2, 3, 4, 5],
        'ue_type': ['device'] * 6,
        'x_pos': [150, 320, 200, 180, 250, 400],
        'y_pos': [230, 450, 300, 120, 350, 100],
        'traffic_type': ['URLLC', 'mMTC', 'mMTC', 'URLLC', 'URLLC', 'mMTC'],
        'priority': [9, 3, 2, 8, 7, 3],
        'data_size': [128, 64, 32, 256, 192, 48]
    })
    initial_states_df.to_csv(initial_states_output, index=False)
    
    # semantic_segments.csv
    semantic_segments_df = pd.DataFrame({
        'ue_id': [0, 1, 1, 2, 3, 3, 4, 4, 5, 5],
        'segment_id': [0, 1, 2, 1, 3, 4, 4, 5, 5, 6]
    })
    semantic_segments_df.to_csv(semantic_segments_output, index=False)
    
    # traffic_model.csv
    traffic_model_data = np.zeros((1000, 6), dtype=int)
    traffic_model_data[0, 0] = 1
    traffic_model_data[1, 1] = 1
    traffic_model_data[2, 2] = 1
    traffic_model_data[3, 0] = 1; traffic_model_data[3, 1] = 1
    traffic_model_data[4, 2] = 1; traffic_model_data[4, 3] = 1
    traffic_model_data[5, 4] = 1; traffic_model_data[5, 5] = 1
    
    traffic_model_df = pd.DataFrame(traffic_model_data, columns=[f'ue_{i}' for i in range(6)])
    traffic_model_df.index.name = 'slot'
    traffic_model_df.to_csv(traffic_model_output)
    
    print("Controlled data files have been created.")

def create_real_time_data():
    """Processes raw data to create the real-time configuration files."""
    file1 = 'data/6G_English_Education_Network_Traffic.csv'
    file2 = 'data/6G_English_Education_Traffic_20204.csv'
    processed_output = 'data/processed_traffic.csv'
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    combined_df = pd.concat([df1, df2], ignore_index=True)

    combined_df = combined_df[combined_df['is_malicious'] == 0]
    combined_df['source_ip'] = combined_df['source_ip'].astype(str)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], format='mixed')
    combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
    
    combined_df.to_csv(processed_output, index=False)
    print(f"Combined data has been preprocessed and saved to {processed_output}")
    print(f"Total records: {len(combined_df)}")

if __name__ == '__main__':
    # Run the data creation for both controlled and real-time environments
    create_controlled_data()
    create_real_time_data()