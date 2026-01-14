import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_all_results(file_path):
    """Loads metrics from a CSV and generates a set of plots."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    df = pd.read_csv(file_path)

    # --- Plot 1: Metrics vs. Episode (Line Plot) ---
    plt.figure(figsize=(15, 10))

    # URLLC Success
    plt.subplot(2, 2, 1)
    plt.plot(df['episode'], df['urllc_success'], label='URLLC Success')
    plt.title('URLLC Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.legend()

    # Collision Rate
    plt.subplot(2, 2, 2)
    plt.plot(df['episode'], df['collision_rate'], label='Collision Rate', color='r')
    plt.title('Collision Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Collision Rate')
    plt.grid(True)
    plt.legend()

    # mMTC Throughput
    plt.subplot(2, 2, 3)
    plt.plot(df['episode'], df['mmtc_throughput'], label='mMTC Throughput', color='g')
    plt.title('mMTC Throughput over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Throughput (packets/slot)')
    plt.grid(True)
    plt.legend()

    # Average Reward
    plt.subplot(2, 2, 4)
    plt.plot(df['episode'], df['reward'], label='Avg Reward', color='purple')
    plt.title('Average Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # Save the first plot
    plt.savefig('outputs/line_plots.png')
    plt.show()

    # --- Plot 2: URLLC Success vs. Collision Rate (Scatter Plot) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(df['collision_rate'], df['urllc_success'])
    plt.title('URLLC Success vs. Collision Rate')
    plt.xlabel('Collision Rate')
    plt.ylabel('URLLC Success Rate')
    plt.grid(True)
    # Save the second plot
    plt.savefig('outputs/scatter_urllc_collision.png')
    plt.show()
    
    # --- Plot 3: URLLC Success vs. mMTC Throughput (Scatter Plot) ---
    plt.figure(figsize=(8, 6))
    plt.scatter(df['mmtc_throughput'], df['urllc_success'])
    plt.title('URLLC Success vs. mMTC Throughput')
    plt.xlabel('mMTC Throughput (packets/slot)')
    plt.ylabel('URLLC Success Rate')
    plt.grid(True)
    # Save the third plot
    plt.savefig('outputs/scatter_urllc_mmtc.png')
    plt.show()


# Assuming your results are saved as 'outputs/metrics.csv'
plot_all_results('outputs/metrics.csv')