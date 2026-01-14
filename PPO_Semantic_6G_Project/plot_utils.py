import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- TensorBoard plotting ---
def plot_from_tensorboard(log_dir, scalar_name, title, output_path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        if not os.path.exists(log_dir):
            print(f"❌ Error in plot_from_tensorboard: {log_dir} does not exist")
            return
        
        event_file = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f])[-1]
        acc = EventAccumulator(event_file)
        acc.Reload()
        
        if scalar_name not in acc.Tags()['scalars']:
            print(f"❌ Warning: Scalar '{scalar_name}' not found in logs.")
            return

        events = acc.Scalars(scalar_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        plt.figure(figsize=(12,6))
        sns.lineplot(x=steps, y=values, color='dodgerblue', linewidth=2)
        plt.title(title, fontsize=16)
        plt.xlabel("Timesteps")
        plt.ylabel("Value")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error in plot_from_tensorboard: {e}")

# --- Final metrics bar chart ---
def plot_final_metrics(metrics_dict, output_path):
    plt.figure(figsize=(10,6))
    df = pd.DataFrame([metrics_dict])
    ax = sns.barplot(data=df)
    ax.bar_label(ax.containers[0], fmt='%.4f', fontsize=10)
    plt.title("Final Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")

# --- Channel utilization pie chart ---
def plot_channel_utilization(action_history, num_channels, output_path):
    try:
        all_actions = np.concatenate(action_history).flatten()
        counts = np.bincount(all_actions, minlength=num_channels)
        labels = [f"Channel {i}" for i in range(num_channels)]
        plt.figure(figsize=(8,8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("viridis", num_channels))
        plt.title("Channel Utilization")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error in plot_channel_utilization: {e}")

# --- Traffic load ---
def plot_traffic_load(traffic_df, output_path):
    try:
        plt.figure(figsize=(15,6))
        sns.lineplot(x=traffic_df.index, y=traffic_df.sum(axis=1), color='firebrick', linewidth=2)
        plt.title("Traffic Load Over Time")
        plt.xlabel("Time Slot")
        plt.ylabel("Active Users")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error in plot_traffic_load: {e}")

# --- Reward components ---
def plot_reward_components(rewards_history, output_path):
    try:
        rewards_history = np.array(rewards_history)
        pos = np.cumsum(np.maximum(0, rewards_history))
        neg = np.cumsum(np.minimum(0, rewards_history))
        plt.figure(figsize=(15,6))
        plt.plot(pos, label='Cumulative Positive', color='forestgreen', linewidth=2)
        plt.plot(neg, label='Cumulative Negative', color='crimson', linewidth=2)
        plt.title("Reward Components")
        plt.xlabel("Time Slot")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error in plot_reward_components: {e}")

# --- Enhanced channel assignments ---
def plot_enhanced_channel_assignments(action_history, traffic_df, initial_states_df, output_path):
    try:
        num_ues = traffic_df.shape[1]
        num_slots = min(len(action_history), len(traffic_df))
        assignment_matrix = np.full((num_ues, num_slots), -1.0)

        for t in range(num_slots):
            active_users = traffic_df.iloc[t]
            for ue_id in range(num_ues):
                if active_users[f'ue_{ue_id}'] == 1:
                    # FIX: MARL actions are 1D arrays, remove [0]
                    assignment_matrix[ue_id, t] = action_history[t][ue_id]

        plt.figure(figsize=(20,8))
        cmap = sns.color_palette("viridis", 3)
        cmap.insert(0, (0.85, 0.85, 0.85)) # inactive
        ax = sns.heatmap(assignment_matrix, cmap=cmap, linewidths=.5, linecolor='lightgray',
                         cbar_kws={'ticks': [-1, 0, 1, 2]})
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['Inactive', 'Channel 0', 'Channel 1', 'Channel 2'])

        num_urllc = initial_states_df[initial_states_df['traffic_type']=='URLLC'].shape[0]
        if 0 < num_urllc < num_ues:
            ax.axhline(y=num_urllc, color='red', linewidth=3, linestyle='--')
            plt.text(num_slots*1.01, num_urllc/2, 'URLLC Users', va='center', ha='left', backgroundcolor='white')
            plt.text(num_slots*1.01, num_urllc + (num_ues-num_urllc)/2, 'mMTC Users', va='center', ha='left', backgroundcolor='white')

        plt.title('Enhanced Channel Assignments')
        plt.xlabel('Time Slot')
        plt.ylabel('UE')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error in plot_enhanced_channel_assignments: {e}")
