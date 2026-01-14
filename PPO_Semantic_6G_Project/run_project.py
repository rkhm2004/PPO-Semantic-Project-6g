import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# --- SAMAEnvironment Class Definition ---
class SAMAEnvironment(gym.Env):
    def __init__(self, is_real_time=False, num_ues=6, num_channels=3, **kwargs):
        super(SAMAEnvironment, self).__init__()
        
        self.is_real_time = is_real_time
        self.num_ues = num_ues
        self.num_channels = num_channels
        
        self.config = self._load_config(kwargs.get('config_path', 'data/channel_params.json'))
        self.initial_states_path = kwargs.get('initial_states_path')
        self.semantic_path = kwargs.get('semantic_path')
        self.traffic_path = kwargs.get('traffic_path')
        
        self.ue_states = self._load_initial_states(self.initial_states_path)
        self.semantic_segments = self._load_semantic_data(self.semantic_path)
        self.traffic_model = self._load_traffic_model(self.traffic_path)
        
        self.max_slots = len(self.traffic_model)
        
        self.observation_space = spaces.Dict({
            "channel_states": spaces.Box(low=0, high=1, shape=(self.num_channels,), dtype=np.int8),
            "ue_active_states": spaces.Box(low=0, high=1, shape=(self.num_ues,), dtype=np.int8),
            "ue_priorities": spaces.Box(low=0.0, high=1.0, shape=(self.num_ues,), dtype=np.float32),
            "semantic_matrix": spaces.Box(low=0, high=1, shape=(self.num_ues, 6), dtype=np.int8)
        })
        
        self.action_space = spaces.MultiDiscrete([self.num_channels] * self.num_ues)
        
        # Initialize state variables
        self.current_slot = 0
        self.channel_states = np.ones(self.num_channels)
        self.collisions = np.zeros(self.num_channels)
        self.spectral_efficiency = 0
        self.urllc_stats = {'success': 0, 'failures': 0}
        self.mmtc_stats = {'throughput': 0, 'packets': 0}
        
    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _load_initial_states(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df['active'] = True
        df['d2lt'] = 0
        return df

    def _load_semantic_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
        
    def _load_traffic_model(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if 'slot' in df.columns:
            df = df.set_index('slot')
        return df
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.ue_states = self._load_initial_states(self.initial_states_path)
        self.semantic_segments = self._load_semantic_data(self.semantic_path)
        self.traffic_model = self._load_traffic_model(self.traffic_path)
        
        self.current_slot = 0
        self.channel_states = np.ones(self.num_channels)
        self.collisions = np.zeros(self.num_channels)
        self.spectral_efficiency = 0
        self.urllc_stats = {'success': 0, 'failures': 0}
        self.mmtc_stats = {'throughput': 0, 'packets': 0}
        
        if self.current_slot < len(self.traffic_model) and self.current_slot in self.traffic_model.index:
            traffic = self.traffic_model.loc[self.current_slot]
            for i in range(self.num_ues):
                self.ue_states.at[i, 'active'] = bool(traffic[f'ue_{i}'])
                
        observation = self._get_observation()
        info = {}
        return observation, info
    
    def _get_observation(self) -> Dict:
        channel_states = self.channel_states.astype(np.int8)
        ue_active_states = np.zeros(self.num_ues, dtype=np.int8)
        ue_priorities = np.zeros(self.num_ues, dtype=np.float32)
        semantic_matrix = np.zeros((self.num_ues, 6), dtype=np.int8)
        
        if self.current_slot < self.max_slots and self.current_slot in self.traffic_model.index:
            traffic = self.traffic_model.loc[self.current_slot]
            for i in range(self.num_ues):
                if traffic[f'ue_{i}'] == 1:
                    ue_active_states[i] = 1
                    ue_priorities[i] = self.ue_states['priority'].iloc[i] / 10.0
                    ue_segments = self.semantic_segments[self.semantic_segments['ue_id'] == i]['segment_id'].tolist()
                    for seg_id in ue_segments:
                        if 0 <= seg_id < 6:
                            semantic_matrix[i, seg_id] = 1
        
        return {
            "channel_states": channel_states,
            "ue_active_states": ue_active_states,
            "ue_priorities": ue_priorities,
            "semantic_matrix": semantic_matrix
        }

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        reward = 0
        channel_usage = defaultdict(list)
        self.ue_states['d2lt'] += 1

        for ue_id, channel in enumerate(action):
            if not (0 <= channel < self.num_channels): continue
            ue = self.ue_states.iloc[ue_id]
            if not ue['active']: continue
            channel_usage[channel].append(ue_id)
        
        successful_segments = set()
        for channel, ue_ids in channel_usage.items():
            if len(ue_ids) > 1:
                self.collisions[channel] += 1
                for ue_id in ue_ids:
                    ue = self.ue_states.iloc[ue_id]
                    reward -= 100 if ue['traffic_type'] == 'URLLC' else 50
            elif len(ue_ids) == 1:
                ue_id = ue_ids[0]
                ue = self.ue_states.iloc[ue_id]

                d2lt_reward_factor = self.ue_states.at[ue_id, 'd2lt'] / self.max_slots
                self.ue_states.at[ue_id, 'd2lt'] = 0

                transmitted_segments = self.semantic_segments[self.semantic_segments['ue_id'] == ue['ue_id']]['segment_id'].tolist()
                
                if ue['traffic_type'] == 'URLLC':
                    self.urllc_stats['success'] += 1
                    reward += (100 + 50 * d2lt_reward_factor)
                else:
                    self.mmtc_stats['packets'] += 1
                    self.mmtc_stats['throughput'] += 1
                    reward += (20 + 10 * d2lt_reward_factor)
                
                for segment in transmitted_segments:
                    if segment not in successful_segments:
                        successful_segments.add(segment)
                        other_ues_with_segment = self.semantic_segments[(self.semantic_segments['ue_id'] != ue_id) & (self.semantic_segments['segment_id'] == segment)]
                        if not other_ues_with_segment.empty:
                            reward += 20

        self.channel_states = np.random.binomial(1, 0.8, size=self.num_channels)
        
        if self.current_slot > 0:
            successful_ues = self.urllc_stats['success'] + self.mmtc_stats['packets']
            self.spectral_efficiency = successful_ues / (self.num_channels * self.current_slot)
        
        self.current_slot += 1

        if self.current_slot < self.max_slots and self.current_slot in self.traffic_model.index:
            traffic = self.traffic_model.loc[self.current_slot]
            for i in range(self.num_ues):
                self.ue_states.at[i, 'active'] = bool(traffic[f'ue_{i}'])

        done = self.current_slot >= self.max_slots
        reward += self.spectral_efficiency * 50
        
        return self._get_observation(), reward, done, False, {}

# --- Data processing and setup scripts ---
def create_controlled_data(initial_path, semantic_path, traffic_path):
    print("Creating controlled data files...")
    initial_states_df = pd.DataFrame({
        'ue_id': [0, 1, 2, 3, 4, 5], 'ue_type': ['device'] * 6,
        'traffic_type': ['URLLC', 'mMTC', 'mMTC', 'URLLC', 'URLLC', 'mMTC'],
        'priority': [9, 3, 2, 8, 7, 3]
    })
    initial_states_df.to_csv(initial_path, index=False)
    
    semantic_segments_df = pd.DataFrame({
        'ue_id': [0, 1, 1, 2, 3, 3, 4, 4, 5, 5],
        'segment_id': [0, 1, 2, 1, 3, 4, 4, 5, 5, 2]
    })
    semantic_segments_df.to_csv(semantic_path, index=False)
    
    traffic_model_df = pd.DataFrame(np.random.randint(0, 2, size=(1000, 6)), columns=[f'ue_{i}' for i in range(6)])
    traffic_model_df.index.name = 'slot'
    traffic_model_df.to_csv(traffic_path)
    print("Controlled data files created.")

def create_real_time_data(processed_output, initial_output, semantic_output, traffic_output):
    print("Processing real-time data files...")
    file1, file2 = 'data/6G_English_Education_Network_Traffic.csv', 'data/6G_English_Education_Traffic_20204.csv'
    
    if not (os.path.exists(file1) and os.path.exists(file2)):
        print(f"Error: Source traffic files not found. Please place them in the 'data' directory.")
        exit()
        
    df = pd.concat([pd.read_csv(file1), pd.read_csv(file2)], ignore_index=True)
    df = df[df['is_malicious'] == 0].copy()
    df['source_ip'] = df['source_ip'].astype(str)
    df.to_csv(processed_output, index=False)
    
    urllc_users = df[df['activity_label'].isin(['quiz', 'VR-session'])]['source_ip'].unique()
    mmtc_users = df[~df['source_ip'].isin(urllc_users)]['source_ip'].unique()
    
    selected_urllc = np.random.choice(urllc_users, size=min(3, len(urllc_users)), replace=False)
    selected_mmtc = np.random.choice(mmtc_users, size=min(3, len(mmtc_users)), replace=False)
    selected_ips = np.concatenate([selected_urllc, selected_mmtc])
    
    initial_states = [{'ue_id': i, 'traffic_type': 'URLLC' if ip in selected_urllc else 'mMTC', 'priority': 9 if ip in selected_urllc else 3} for i, ip in enumerate(selected_ips)]
    pd.DataFrame(initial_states).to_csv(initial_output, index=False)
    
    activity_map = {'discussion': 0, 'stream': 1, 'quiz': 2, 'login': 3, 'VR-session': 4, 'submit_assignment': 5}
    semantic_data = []
    for ue_id, ip in enumerate(selected_ips):
        activities = df[df['source_ip'] == ip]['activity_label'].unique()
        for act in activities:
            if act in activity_map:
                semantic_data.append({'ue_id': ue_id, 'segment_id': activity_map[act]})
    pd.DataFrame(semantic_data).to_csv(semantic_output, index=False)

    ip_map = {ip: i for i, ip in enumerate(selected_ips)}
    traffic_df = pd.DataFrame(0, index=range(len(df)), columns=[f'ue_{i}' for i in range(len(selected_ips))])
    for i, row in df.iterrows():
        ue_id = ip_map.get(row['source_ip'])
        if ue_id is not None:
            traffic_df.loc[i, f'ue_{ue_id}'] = 1
    traffic_df.index.name = 'slot'
    traffic_df.to_csv(traffic_output)
    print("Real-time environment configuration files created.")

def create_channel_params():
    os.makedirs('data', exist_ok=True)
    params = {"bandwidth": 100e6, "num_channels": 3}
    with open('data/channel_params.json', 'w') as f:
        json.dump(params, f, indent=4)
    print("channel_params.json created.")

def run_training_pipeline(is_real_time):
    os.makedirs('outputs', exist_ok=True)
    
    if is_real_time:
        os.makedirs('data/real_time', exist_ok=True)
        paths = {'initial': 'data/real_time/initial_states.csv', 'semantic': 'data/real_time/semantic_segments.csv', 'traffic': 'data/real_time/traffic_model.csv'}
        create_real_time_data('data/processed_traffic.csv', paths['initial'], paths['semantic'], paths['traffic'])
        model_name = "ppo_real_time"
    else:
        os.makedirs('data/controlled', exist_ok=True)
        paths = {'initial': 'data/controlled/initial_states.csv', 'semantic': 'data/controlled/semantic_segments.csv', 'traffic': 'data/controlled/traffic_model.csv'}
        create_controlled_data(paths['initial'], paths['semantic'], paths['traffic'])
        model_name = "ppo_controlled"
    
    # DYNAMICALLY get the number of users from the data file
    initial_states_df = pd.read_csv(paths['initial'])
    actual_num_ues = len(initial_states_df)
    print(f"✅ Training with {actual_num_ues} users detected from data.")

    env_config = {
        'is_real_time': is_real_time,
        'num_ues': actual_num_ues,
        'initial_states_path': paths['initial'],
        'semantic_path': paths['semantic'],
        'traffic_path': paths['traffic']
    }
    
    env = DummyVecEnv([lambda: SAMAEnvironment(**env_config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    model = PPO("MultiInputPolicy", env, verbose=1, n_steps=1024, learning_rate=0.0003, gamma=0.99)
    
    print(f"\n🚀 Starting training for '{model_name}' environment...")
    model.learn(total_timesteps=20000, progress_bar=True)
    
    model_path = f"outputs/{model_name}"
    model.save(model_path)
    print(f"\n🎉 Training completed. Model saved to '{model_path}'")

if __name__ == '__main__':
    create_channel_params()
    is_real_time = True # Set to True to use real traffic data
    # CORRECTED: Removed stray text from the function call
    run_training_pipeline(is_real_time)