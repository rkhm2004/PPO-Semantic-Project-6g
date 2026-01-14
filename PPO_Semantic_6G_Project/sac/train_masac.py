import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from collections import deque, namedtuple
import random

from ma_environment_sac import MASAC_SAMAEnvironment
from ma_policies_sac import DecentralizedActorSAC, CentralizedCriticSAC

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
        self.Experience = namedtuple('Experience', field_names=['global_state', 'obs', 'actions', 'reward', 'next_global_state', 'done'])
    
    def add(self, global_state, obs, actions, reward, next_global_state, done):
        e = self.Experience(global_state, obs, actions, reward, next_global_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size, agent_ids):
        experiences = random.sample(self.buffer, k=batch_size)
        global_states = torch.FloatTensor(np.array([e.global_state for e in experiences]))
        obs = {agent: torch.FloatTensor(np.array([e.obs[agent] for e in experiences])) for agent in agent_ids}
        actions = {agent: torch.FloatTensor(np.array([e.actions[agent] for e in experiences])) for agent in agent_ids}
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1)
        next_global_states = torch.FloatTensor(np.array([e.next_global_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1)
        return global_states, obs, actions, rewards, next_global_states, dones

    def __len__(self):
        return len(self.buffer)

def train_masac(num_episodes=10, lr=3e-4, gamma=0.99, tau=0.005, batch_size=256, buffer_size=1e6):
    print("--- Starting Multi-Agent SAC (MASAC) Training ---")
    
    # --- CORRECTED: Use robust paths in env_config ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data')
    
    env_config = {
        'is_real_time': True, 'num_ues': 6, 'num_channels': 3,
        'state_path': os.path.join(data_path, 'real/real_initial_states.csv'),
        'semantic_path': os.path.join(data_path, 'real/real_semantic_segments.csv'),
        'traffic_path': os.path.join(data_path, 'real/real_traffic_model.csv'),
        'config_path': os.path.join(data_path, 'channel_params_sac.json')
    }
    env = MASAC_SAMAEnvironment(**env_config)
    
    num_agents = len(env.possible_agents)
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    
    actors = {agent_id: DecentralizedActorSAC(obs_dim, action_dim) for agent_id in agent_ids}
    actor_optimizers = {agent_id: optim.Adam(actor.parameters(), lr=lr) for agent_id, actor in actors.items()}
    
    critic = CentralizedCriticSAC(num_agents, obs_dim, action_dim)
    critic_target = CentralizedCriticSAC(num_agents, obs_dim, action_dim)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)
    
    target_entropy = -torch.prod(torch.Tensor(env.action_space(agent_ids[0]).shape)).item()
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_optimizer = optim.Adam([log_alpha], lr=lr)
    
    replay_buffer = ReplayBuffer(buffer_size)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        ep_metrics = {'urllc_success_rate': [], 'mmtc_throughput': [], 'collision_rate': []}

        while not done:
            global_state = list(obs.values())[0]
            actions = {}
            with torch.no_grad():
                for agent_id, agent_obs in obs.items():
                    action, _ = actors[agent_id].get_action(torch.FloatTensor(agent_obs).unsqueeze(0))
                    actions[agent_id] = action.numpy()[0]
            
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            
            done = any(terminations.values()) or any(truncations.values())
            replay_buffer.add(global_state, obs, actions, rewards[agent_ids[0]], list(next_obs.values())[0], done)
            
            obs = next_obs
            episode_reward += rewards[agent_ids[0]]
            
            info_sample = infos.get(agent_ids[0], {})
            for key in ep_metrics.keys():
                if key in info_sample: ep_metrics[key].append(info_sample[key])

            if len(replay_buffer) > batch_size:
                g_states, b_obs, b_actions, b_rewards, next_g_states, b_dones = replay_buffer.sample(batch_size, agent_ids)

                with torch.no_grad():
                    next_actions, next_log_probs = {}, {}
                    for agent_id in agent_ids:
                        a, lp = actors[agent_id].get_action(b_obs[agent_id])
                        next_actions[agent_id] = a
                        next_log_probs[agent_id] = lp
                    
                    next_actions_tensor = torch.cat(list(next_actions.values()), dim=-1)
                    next_log_probs_tensor = torch.stack(list(next_log_probs.values()), dim=0).sum(0)
                    
                    q1_target, q2_target = critic_target(next_g_states, next_actions_tensor)
                    q_target = torch.min(q1_target, q2_target) - log_alpha.exp() * next_log_probs_tensor.unsqueeze(1)
                    target = b_rewards + gamma * (1 - b_dones) * q_target
                
                q1, q2 = critic(g_states, torch.cat(list(b_actions.values()), dim=-1))
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                new_actions, new_log_probs = {}, {}
                for agent_id in agent_ids:
                    a, lp = actors[agent_id].get_action(b_obs[agent_id])
                    new_actions[agent_id] = a
                    new_log_probs[agent_id] = lp
                
                new_actions_tensor = torch.cat(list(new_actions.values()), dim=-1)
                new_log_probs_tensor = torch.stack(list(new_log_probs.values()), dim=0).sum(0)
                
                q1_pred, q2_pred = critic(g_states, new_actions_tensor)
                q_pred = torch.min(q1_pred, q2_pred)
                
                actor_loss = ((log_alpha.exp() * new_log_probs_tensor) - q_pred.view(-1)).mean()
                
                for opt in actor_optimizers.values(): opt.zero_grad()
                actor_loss.backward()
                for opt in actor_optimizers.values(): opt.step()
                
                alpha_loss = (-log_alpha * (new_log_probs_tensor + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        if (episode + 1) % 10 == 0:
            avg_metrics = {key: np.mean(values) for key, values in ep_metrics.items() if values}
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f} | "
                  f"URLLC: {avg_metrics.get('urllc_success_rate', 0):.4f} | "
                  f"mMTC: {avg_metrics.get('mmtc_throughput', 0):.4f}")

    print("\n--- MASAC Training Finished ---")

if __name__ == '__main__':
    train_masac()