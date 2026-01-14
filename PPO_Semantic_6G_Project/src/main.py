import numpy as np
from environment import SAMAEnvironment
from agent import PPOScheduler
from utils import save_results
import time
import os

def main():
    # Create output directories
    os.makedirs('outputs/logs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Initialize environment and agent
    env = SAMAEnvironment()
    state = env.reset()
    
    # Calculate state and action dimensions
    state_dim = len(state)
    action_dim = env.num_channels
    num_ues = len(env.ue_states)

    # Training parameters
    num_episodes = 5
    max_steps = 500000
    save_interval = 50
    
    # New hyperparameters to improve training
    learning_rate = 0.0001
    gamma = 0.999
    
    agent = PPOScheduler(state_dim, action_dim, num_ues, lr=learning_rate, gamma=gamma)
    
    # Metrics tracking
    metrics = {
        'episode': [],
        'reward': [],
        'urllc_success': [],
        'mmtc_throughput': [],
        'spectral_eff': [],
        'collision_rate': []
    }
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_metrics = {
            'urllc_success': [],
            'mmtc_throughput': [],
            'spectral_eff': [],
            'collision_rate': []
        }
        
        for step in range(max_steps):
            # Get action from agent
            action, _ = agent.get_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            episode_metrics['urllc_success'].append(info['urllc_success_rate'])
            episode_metrics['mmtc_throughput'].append(info['mmtc_throughput'])
            episode_metrics['spectral_eff'].append(info['spectral_efficiency'])
            episode_metrics['collision_rate'].append(info['collision_rate'])
            
            # Update agent
            agent.update()
            
            state = next_state
            if done:
                break
        
        # Calculate average metrics for the episode
        avg_urllc = np.mean(episode_metrics['urllc_success'])
        avg_mmtc = np.mean(episode_metrics['mmtc_throughput'])
        avg_spectral = np.mean(episode_metrics['spectral_eff'])
        avg_collision = np.mean(episode_metrics['collision_rate'])
        
        # Store metrics
        metrics['episode'].append(episode)
        metrics['reward'].append(episode_reward / max_steps)
        metrics['urllc_success'].append(avg_urllc)
        metrics['mmtc_throughput'].append(avg_mmtc)
        metrics['spectral_eff'].append(avg_spectral)
        metrics['collision_rate'].append(avg_collision)
        
        # Print progress
        print(f"Episode {episode}/{num_episodes}:")
        print(f"  Avg Reward: {episode_reward / max_steps:.2f}")
        print(f"  URLLC Success: {avg_urllc:.4f}")
        print(f"  mMTC Throughput: {avg_mmtc:.2f} packets/slot")
        print(f"  Spectral Efficiency: {avg_spectral:.4f}")
        print(f"  Collision Rate: {avg_collision:.4f}")
        
        # Save models and metrics periodically
        if episode % save_interval == 0:
            agent.save_models(f'outputs/models_episode_{episode}.pt')
            save_results(metrics, 'outputs')
    
    # Save final models and metrics
    agent.save_models('outputs/final_model.pt')
    save_results(metrics, 'outputs')
    print("Training completed!")

if __name__ == "__main__":
    main()