import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List
import random
from collections import defaultdict

class SAMAEnvironment:
    def __init__(self, config_path: str = 'data/channel_params.json', 
                 state_path: str = 'data/initial_states.csv',
                 semantic_path: str = 'data/semantic_segments.csv',
                 traffic_path: str = 'data/traffic_model.csv'):
        self.config = self._load_config(config_path)
        self.ue_states = self._load_initial_states(state_path)
        self.semantic_segments = self._load_semantic_data(semantic_path)
        self.traffic_model = self._load_traffic_model(traffic_path)
        self.num_channels = self.config['num_channels']
        self.current_slot = 0
        self.max_slots = 1000
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
        df['d2lt'] = 0  # Add D2LT metric, initialized to 0
        return df

    def _load_semantic_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
        
    def _load_traffic_model(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.set_index('slot')
        return df
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_slot = 0
        self.channel_states = np.ones(self.num_channels)
        self.collisions = np.zeros(self.num_channels)
        self.spectral_efficiency = 0
        self.urllc_stats = {'success': 0, 'failures': 0}
        self.mmtc_stats = {'throughput': 0, 'packets': 0}
        self.ue_states['d2lt'] = 0 # Reset D2LT for all UEs
        # Apply initial traffic pattern from the model
        if self.current_slot in self.traffic_model.index:
            traffic = self.traffic_model.loc[self.current_slot]
            self.ue_states['active'] = [bool(traffic[f'ue_{i}']) for i in range(len(self.ue_states))]
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current environment's global state"""
        state = []
        
        # Channel states (occupied/free)
        state.extend(self.channel_states)
        
        # Normalize D2LT for state
        total_d2lt = self.ue_states['d2lt'].sum()
        if total_d2lt == 0:
            normalized_d2lt = np.zeros(len(self.ue_states))
        else:
            normalized_d2lt = self.ue_states['d2lt'] / total_d2lt

        # UE information (traffic type, priority, semantic segments, and D2LT)
        for idx, ue in self.ue_states.iterrows():
            state.append(1 if ue['active'] else 0)
            state.append(1 if ue['traffic_type'] == 'URLLC' else 0)
            state.append(ue['priority'] / 10)
            state.append(normalized_d2lt[idx])  # Add normalized D2LT to state
            
            # Semantic segments for the UE
            ue_segments = self.semantic_segments[self.semantic_segments['ue_id'] == ue['ue_id']]['segment_id'].tolist()
            segment_vector = [1 if seg in ue_segments else 0 for seg in range(10)]
            state.extend(segment_vector)
            
        # Network performance metrics
        state.append(self.spectral_efficiency)
        state.append(self.urllc_stats['success'] / (self.urllc_stats['success'] + self.urllc_stats['failures'] + 1e-6))
        state.append(self.mmtc_stats['throughput'] / (self.mmtc_stats['packets'] + 1e-6))
        
        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time slot in the environment
        """
        reward = 0
        channel_usage = defaultdict(list)
        
        # Increment D2LT for all UEs at the start of the step
        self.ue_states['d2lt'] += 1

        for ue_id, channel in enumerate(action):
            if channel < 0 or channel >= self.num_channels:
                continue
            
            ue = self.ue_states.iloc[ue_id]
            if not ue['active']:
                continue
                
            channel_usage[channel].append(ue_id)
        
        successful_segments = set()
        
        for channel, ue_ids in channel_usage.items():
            if len(ue_ids) > 1:
                # Collision occurred on this channel
                self.collisions[channel] += 1
                for ue_id in ue_ids:
                    ue = self.ue_states.iloc[ue_id]
                    if ue['traffic_type'] == 'URLLC':
                        self.urllc_stats['failures'] += 1
                        # --- FIX: Significantly increased penalty for URLLC failure ---
                        reward -= 100  
                    else:
                        # --- FIX: Significantly increased penalty for mMTC failure ---
                        reward -= 50 
            else:
                # Successful transmission
                ue_id = ue_ids[0]
                ue = self.ue_states.iloc[ue_id]
                
                # Reset D2LT for the successful UE
                self.ue_states.at[ue_id, 'd2lt'] = 0

                # Get the segments associated with the successful UE
                transmitted_segments = self.semantic_segments[self.semantic_segments['ue_id'] == ue['ue_id']]['segment_id'].tolist()
                
                # Reward based on D2LT
                d2lt_reward_factor = self.ue_states.at[ue_id, 'd2lt'] / self.max_slots
                if ue['traffic_type'] == 'URLLC':
                    self.urllc_stats['success'] += 1
                    # --- FIX: Increased reward for URLLC success ---
                    reward += (100 + 50 * d2lt_reward_factor)
                else:
                    self.mmtc_stats['packets'] += 1
                    self.mmtc_stats['throughput'] += 1
                    # A moderate reward for mMTC success
                    reward += (10 + 10 * d2lt_reward_factor)
                
                for segment in transmitted_segments:
                    if segment not in successful_segments:
                        successful_segments.add(segment)
                        other_ues_with_segment = self.semantic_segments[(self.semantic_segments['ue_id'] != ue_id) & (self.semantic_segments['segment_id'] == segment)]
                        if not other_ues_with_segment.empty:
                            reward += 20
        
        # Update channel states
        self.channel_states = np.random.binomial(1, 0.8, size=self.num_channels)
        
        # Calculate spectral efficiency
        total_ues = len(self.ue_states)
        successful_ues = self.urllc_stats['success'] + self.mmtc_stats['packets']
        self.spectral_efficiency = successful_ues / (self.num_channels * (self.current_slot + 1))
        
        # Update traffic based on the new model
        if self.current_slot < len(self.traffic_model) and self.current_slot in self.traffic_model.index:
            traffic = self.traffic_model.loc[self.current_slot]
            for i in range(len(self.ue_states)):
                self.ue_states.at[i, 'active'] = bool(traffic[f'ue_{i}'])
        
        self.current_slot += 1
        done = self.current_slot >= self.max_slots
        
        reward += self.spectral_efficiency * 50
        
        return self._get_state(), reward, done, {
            'urllc_success_rate': self.urllc_stats['success'] / (self.urllc_stats['success'] + self.urllc_stats['failures'] + 1e-6),
            'mmtc_throughput': self.mmtc_stats['throughput'] / (self.current_slot + 1),
            'spectral_efficiency': self.spectral_efficiency,
            'collision_rate': np.mean(self.collisions) / (self.current_slot + 1)
        }
    
    def _generate_traffic(self):
        """This function is no longer used, as traffic is now loaded from a file."""
        pass