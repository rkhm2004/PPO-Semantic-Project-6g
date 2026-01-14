import torch
import pandas as pd
import numpy as np
import os
import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from SAMAEnvironment_sac import SAMAEnvironment_SAC
from visualize import (
    plot_from_tensorboard, plot_final_metrics, plot_channel_utilization,
    plot_traffic_load, plot_reward_components, plot_enhanced_channel_assignments
)

def run_training_pipeline(env_config, model_name, output_dir):
    print(f"Starting training for '{model_name}' using SAC...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tensorboard_log_dir = os.path.join(output_dir, "tensorboard_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    env = DummyVecEnv([lambda: SAMAEnvironment_SAC(**env_config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=50000,
        batch_size=256,
        gamma=0.99,
        tensorboard_log=tensorboard_log_dir,
        device=device
    )

    model.learn(total_timesteps=30000, progress_bar=True, log_interval=4)
    
    model_path = os.path.join(output_dir, model_name)
    model.save(model_path)
    env.save(os.path.join(output_dir, "vec_normalize_sac.pkl"))
    print(f"Training completed and model saved to '{model_path}'")

    print("\nRunning final evaluation...")
    obs = env.reset()
    dones = [False]
    action_history, rewards_history = [], []
    urllc_s, mmtc_t, coll_r, spec_e = [], [], [], []

    traffic_df = pd.read_csv(env_config['traffic_path'], index_col='slot')
    for _ in range(len(traffic_df)):
        action, _ = model.predict(obs, deterministic=True)
        # Convert continuous action to discrete for plotting history
        discrete_action = np.round(action).astype(int)
        discrete_action = np.clip(discrete_action, 0, env_config['num_channels'] - 1)
        action_history.append(discrete_action)
        
        obs, rewards, dones, infos = env.step(action)
        rewards_history.append(rewards[0])
        info = infos[0]
        urllc_s.append(info['urllc_success_rate'])
        mmtc_t.append(info['mmtc_throughput'])
        spec_e.append(info['spectral_efficiency'])
        coll_r.append(info.get('collision_rate', 0))
        if dones[0]:
            break
            
    print("\n--- Generating All Visualizations ---")
    
    latest_run_log_dir = sorted([os.path.join(tensorboard_log_dir, d) for d in os.listdir(tensorboard_log_dir)])[-1]
    plot_from_tensorboard(latest_run_log_dir, 'rollout/ep_rew_mean', 'Learning Curve (Mean Reward)', os.path.join(output_dir, '1_learning_curve.png'))
    
    final_metrics = {
        "URLLC Success": np.mean(urllc_s), "mMTC Throughput": np.mean(mmtc_t),
        "Collision Rate": np.mean(coll_r), "Spectral Efficiency": np.mean(spec_e)
    }
    plot_final_metrics(final_metrics, os.path.join(output_dir, '3_final_metrics_barchart.png'))
    plot_channel_utilization(action_history, env_config['num_channels'], os.path.join(output_dir, '4_channel_utilization_piechart.png'))
    plot_traffic_load(traffic_df, os.path.join(output_dir, '5_traffic_load_over_time.png'))
    plot_reward_components(rewards_history, os.path.join(output_dir, '6_reward_components.png'))
    
    initial_states_df = pd.read_csv(env_config['state_path'])
    # The action history needs to be reshaped slightly for the heatmap
    plot_enhanced_channel_assignments([a.reshape(1, -1) for a in action_history], traffic_df, initial_states_df, os.path.join(output_dir, '7_enhanced_channel_assignments.png'))
    
    print("\n--- All visualizations have been saved! ---")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join("outputs", "output_sac", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs for this run will be saved in: {output_dir}")

    real_time_initial_path = 'data/real/real_initial_states.csv'

    try:
        initial_states_df = pd.read_csv(real_time_initial_path)
        actual_num_ues = len(initial_states_df)
        print(f"✅ Detected {actual_num_ues} UEs from the data file.")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at '{real_time_initial_path}'. Please run the data setup script first.")
        exit()

    real_time_env_config = {
        'is_real_time': True, 'num_ues': actual_num_ues, 'num_channels': 3,
        'state_path': real_time_initial_path, 'semantic_path': 'data/real/real_semantic_segments.csv',
        'traffic_path': 'data/real/real_traffic_model.csv', 'config_path': 'data/channel_params.json'
    }

    run_training_pipeline(real_time_env_config, "sac_real_time", output_dir)