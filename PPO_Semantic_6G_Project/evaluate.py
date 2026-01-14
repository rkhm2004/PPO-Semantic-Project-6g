import torch
import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from SAMAEnvironment import SAMAEnvironment

# Import All Plotting Functions
from visualize import (
    plot_final_metrics,
    plot_channel_utilization,
    plot_traffic_load,
    plot_reward_components,
    plot_enhanced_channel_assignments
)

def run_evaluation(num_episodes=10):
    """
    Loads a trained model and evaluates it, printing episodic logs and generating plots.
    """
    model_name = "ppo_real_time"
    model_path = f"outputs/{model_name}.zip"
    vec_normalize_stats_path = "outputs/vec_normalize.pkl"
    output_dir = "outputs/output_6g"
    
    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading model from: {model_path}")
    print(f"All visualizations will be saved in: {output_dir}")

    # --- Load Environment Configuration ---
    initial_states_path = 'data/real/real_initial_states.csv'
    try:
        initial_states_df = pd.read_csv(initial_states_path)
        actual_num_ues = len(initial_states_df)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at '{initial_states_path}'. Please run the data setup script first.")
        return

    env_config = {
        'is_real_time': True, 'num_ues': actual_num_ues,
        'state_path': initial_states_path, 'semantic_path': 'data/real/real_semantic_segments.csv',
        'traffic_path': 'data/real/real_traffic_model.csv', 'config_path': 'data/channel_params.json'
    }

    # --- Load and Wrap Environment ---
    env = DummyVecEnv([lambda: SAMAEnvironment(**env_config)])
    env = VecNormalize.load(vec_normalize_stats_path, env)
    env.training = False # Set to evaluation mode
    env.norm_reward = False

    # --- Load the Trained Model ---
    model = PPO.load(model_path, env=env)

    # --- Run Evaluation Episodes ---
    print("\n--- Running Evaluation Episodes ---")
    for i in range(num_episodes):
        obs = env.reset()
        dones = [False]
        action_history, rewards_history = [], []
        urllc, mmtc, collisions, spectral = [], [], [], []

        traffic_df = pd.read_csv(env_config['traffic_path'], index_col='slot')
        while not dones[0]:
            action, _ = model.predict(obs, deterministic=True)
            action_history.append(action)
            obs, rewards, dones, infos = env.step(action)
            rewards_history.append(rewards[0])
            info = infos[0]
            
            urllc.append(info['urllc_success_rate'])
            mmtc.append(info['mmtc_throughput'])
            spectral.append(info['spectral_efficiency'])
            collisions.append(info.get('collision_rate', 0))
        
        # --- Print Episodic Log ---
        print(f"\nEpisode {i + 1}/{num_episodes}:")
        print(f"  Total Reward: {np.sum(rewards_history):.2f}")
        print(f"  Avg URLLC Success: {np.mean(urllc):.4f}")
        print(f"  Avg mMTC Throughput: {np.mean(mmtc):.4f} packets/slot")
        print(f"  Avg Collision Rate: {np.mean(collisions):.4f}")
        
    # --- Generate Visualizations from the last episode ---
    print("\n\n--- Generating All Visualizations ---")
    
    final_metrics = {
        "URLLC Success": np.mean(urllc), "mMTC Throughput": np.mean(mmtc),
        "Collision Rate": np.mean(collisions), "Spectral Efficiency": np.mean(spectral)
    }
    plot_final_metrics(final_metrics, os.path.join(output_dir, '1_final_metrics_barchart.png'))
    plot_channel_utilization(action_history, env.action_space.nvec[0], os.path.join(output_dir, '2_channel_utilization_piechart.png'))
    plot_traffic_load(traffic_df, os.path.join(output_dir, '3_traffic_load_over_time.png'))
    plot_reward_components(rewards_history, os.path.join(output_dir, '4_reward_components.png'))
    plot_enhanced_channel_assignments(action_history, traffic_df, initial_states_df, os.path.join(output_dir, '5_enhanced_channel_assignments.png'))
    
    print("\n--- All visualizations have been saved! ---")

if __name__ == '__main__':
    run_evaluation()