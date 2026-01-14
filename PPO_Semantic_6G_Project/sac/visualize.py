import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_from_tensorboard(log_dir, scalar_name, title, output_path):
    """Reads a scalar from TensorBoard logs and plots it."""
    print(f"Generating plot for: {title}...")
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        event_file = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f])[-1]
        
        acc = EventAccumulator(event_file)
        acc.Reload()
        
        if scalar_name not in acc.Tags()['scalars']:
            print(f"  > Warning: Scalar '{scalar_name}' not found in logs. Skipping plot.")
            return

        events = acc.Scalars(scalar_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=steps, y=values, color='dodgerblue', linewidth=2)
        plt.title(title, fontsize=16)
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"  > Saved to {os.path.basename(output_path)}")

    except Exception as e:
        print(f"  > Could not generate plot for {scalar_name}. Error: {e}")

def plot_final_metrics(metrics_dict, output_path):
    """Generates a bar chart of the final evaluation metrics."""
    print("Generating plot for: Final Metrics...")
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame([metrics_dict])
    ax = sns.barplot(data=metrics_df)
    ax.bar_label(ax.containers[0], fmt='%.4f', fontsize=10, color='black')
    plt.title("Final Evaluation Metrics", fontsize=16)
    plt.ylabel("Average Score", fontsize=12)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  > Saved to {os.path.basename(output_path)}")

def plot_channel_utilization(action_history, num_channels, output_path):
    """Generates a pie chart of channel utilization."""
    print("Generating plot for: Channel Utilization...")
    all_actions = np.concatenate(action_history).flatten()
    channel_counts = np.bincount(all_actions, minlength=num_channels)
    
    labels = [f'Channel {i}' for i in range(num_channels)]
    
    plt.figure(figsize=(8, 8))
    plt.pie(channel_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", num_channels))
    plt.title("Channel Utilization During Evaluation", fontsize=16)
    plt.savefig(output_path)
    plt.close()
    print(f"  > Saved to {os.path.basename(output_path)}")

def plot_enhanced_channel_assignments(action_history, traffic_df, initial_states_df, output_path):
    """Generates an enhanced heatmap with separators for URLLC and mMTC users."""
    print("Generating plot for: Enhanced Channel Assignments...")
    num_ues = traffic_df.shape[1]
    num_slots = min(len(action_history), len(traffic_df))
    assignment_matrix = np.full((num_ues, num_slots), -1.0)

    for t in range(num_slots):
        active_users_at_t = traffic_df.iloc[t]
        for ue_id in range(num_ues):
            if active_users_at_t[f'ue_{ue_id}'] == 1:
                assignment_matrix[ue_id, t] = action_history[t][0][ue_id]

    plt.figure(figsize=(20, 8))
    cmap = sns.color_palette("viridis", 3)
    cmap.insert(0, (0.85, 0.85, 0.85)) # Lighter gray for inactive

    ax = sns.heatmap(
        assignment_matrix,
        cmap=cmap,
        cbar_kws={'ticks': [-1, 0, 1, 2]},
        linewidths=.5,
        linecolor='lightgray'
    )
    cbar = ax.collections[0].colorbar
    cbar.set_ticklabels(['Inactive', 'Channel 0', 'Channel 1', 'Channel 2'])
    
    num_urllc = initial_states_df[initial_states_df['traffic_type'] == 'URLLC'].shape[0]
    if 0 < num_urllc < num_ues:
        ax.axhline(y=num_urllc, color='red', linewidth=3, linestyle='--')
        plt.text(num_slots * 1.01, num_urllc / 2, 'URLLC Users', va='center', ha='left', backgroundcolor='white')
        plt.text(num_slots * 1.01, num_urllc + (num_ues - num_urllc) / 2, 'mMTC Users', va='center', ha='left', backgroundcolor='white')

    plt.title('Enhanced Channel Assignments (URLLC vs. mMTC)', fontsize=16)
    plt.xlabel('Time Slot', fontsize=12)
    plt.ylabel('User Equipment (UE)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  > Saved to {os.path.basename(output_path)}")

def plot_traffic_load(traffic_df, output_path):
    """Plots the number of active users over time to show traffic load."""
    print("Generating plot for: Traffic Load...")
    active_users_per_slot = traffic_df.sum(axis=1)
    
    plt.figure(figsize=(15, 6))
    sns.lineplot(x=active_users_per_slot.index, y=active_users_per_slot.values, color='firebrick', linewidth=2)
    plt.title("Traffic Load Over Time", fontsize=16)
    plt.xlabel("Time Slot", fontsize=12)
    plt.ylabel("Number of Active Users", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  > Saved to {os.path.basename(output_path)}")

def plot_reward_components(rewards_history, output_path):
    """Plots the cumulative positive and negative rewards over the evaluation episode."""
    print("Generating plot for: Reward Components...")
    rewards_history = np.array(rewards_history)
    positive_rewards = np.maximum(0, rewards_history)
    negative_rewards = np.minimum(0, rewards_history)
    
    cum_positive = np.cumsum(positive_rewards)
    cum_negative = np.cumsum(negative_rewards)
    
    plt.figure(figsize=(15, 6))
    plt.plot(cum_positive, label='Cumulative Positive Rewards (Success)', color='forestgreen', linewidth=2)
    plt.plot(cum_negative, label='Cumulative Negative Rewards (Collisions)', color='crimson', linewidth=2)
    plt.title("Reward Components During Evaluation", fontsize=16)
    plt.xlabel("Time Slot", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  > Saved to {os.path.basename(output_path)}")