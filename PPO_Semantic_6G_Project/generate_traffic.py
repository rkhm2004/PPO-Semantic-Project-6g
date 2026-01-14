import numpy as np
import pandas as pd
import csv

def generate_traffic_model(num_ues, urllc_count, mmtc_count, num_slots):
    """
    Generates a synthetic traffic model with a mix of URLLC (Poisson) and mMTC (Pareto) traffic.
    """
    # Create the header for the CSV file
    header = ['slot'] + [f'ue_{i}' for i in range(num_ues)]
    
    # Initialize the data with zeros
    traffic_data = np.zeros((num_slots, num_ues), dtype=int)
    
    # -- Generate URLLC Traffic (Poisson Distribution) --
    # These UEs have a consistent, low-latency traffic pattern
    urllc_ues = np.random.choice(range(num_ues), size=urllc_count, replace=False)
    for ue_id in urllc_ues:
        # Poisson distribution for consistent, event-driven traffic (e.g., 10 events per 100 slots)
        poisson_lambda = num_slots / 10
        # This will create a list of timestamps for when traffic occurs
        urllc_events = np.random.poisson(poisson_lambda, size=num_slots)
        for t in range(num_slots):
            if urllc_events[t] > 0:
                traffic_data[t, ue_id] = 1

    # -- Generate mMTC Traffic (Pareto Distribution) --
    # These UEs have bursty traffic (long periods of silence with sudden activity)
    mmtc_ues = np.random.choice(list(set(range(num_ues)) - set(urllc_ues)), size=mmtc_count, replace=False)
    for ue_id in mmtc_ues:
        # Pareto distribution for bursty, heavy-tailed traffic
        pareto_alpha = 1.5 # Controls the "burstiness" of the traffic
        pareto_bursts = np.random.pareto(pareto_alpha, num_slots // 10) # Generate some bursts
        for t, burst_length in enumerate(pareto_bursts.astype(int)):
            start_slot = t * 10
            for i in range(min(burst_length, 10)): # Limit bursts to 10 slots
                if start_slot + i < num_slots:
                    traffic_data[start_slot + i, ue_id] = 1

    # Write the data to a CSV file
    with open('data/traffic_model.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(num_slots):
            writer.writerow([i] + list(traffic_data[i]))

# --- Parameters for your project ---
NUM_UES = 6
NUM_URLLC = 3
NUM_MMTC = 3
NUM_SLOTS = 1000 # Increase this for a longer simulation

generate_traffic_model(NUM_UES, NUM_URLLC, NUM_MMTC, NUM_SLOTS)
print("Realistic traffic_model.csv has been created in the 'data' directory.")