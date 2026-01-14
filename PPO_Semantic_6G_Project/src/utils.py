import csv
import matplotlib.pyplot as plt
import os
from typing import Dict, List
import numpy as np

def save_results(metrics: Dict[str, List], output_dir: str = 'outputs'):
    """Save training metrics to CSV and plot graphs"""
    # Create directories if they don't exist
    os.makedirs(f'{output_dir}/plots', exist_ok=True)
    
    # Save to CSV
    with open(f'{output_dir}/metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerows(zip(*metrics.values()))
    
    # Create subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Reward and Spectral Efficiency
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode'], metrics['reward'], 'b-', label='Reward')
    plt.plot(metrics['episode'], np.array(metrics['spectral_eff']) * 100, 'g--', label='Spectral Eff (%)')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Reward and Spectral Efficiency')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: URLLC Success Rate
    plt.subplot(2, 2, 2)
    plt.plot(metrics['episode'], metrics['urllc_success'], 'r-')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('URLLC Success Rate')
    plt.ylim(0.8, 1.05)
    plt.grid(True)
    
    # Plot 3: mMTC Throughput
    plt.subplot(2, 2, 3)
    plt.plot(metrics['episode'], metrics['mmtc_throughput'], 'm-')
    plt.xlabel('Episode')
    plt.ylabel('Packets/Slot')
    plt.title('mMTC Throughput')
    plt.ylim(0.7, 1.2)
    plt.grid(True)
    
    # Plot 4: Collision Rate
    plt.subplot(2, 2, 4)
    plt.plot(metrics['episode'], metrics['collision_rate'], 'k-')
    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.title('Collision Rate')
    plt.ylim(0, 0.02)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual plots
    save_individual_plots(metrics, output_dir)

def save_individual_plots(metrics: Dict[str, List], output_dir: str):
    """Save each metric as a separate plot"""
    metrics_to_plot = {
        'reward': ('Reward', 'b-'),
        'urllc_success': ('URLLC Success Rate', 'r-'),
        'mmtc_throughput': ('mMTC Throughput (packets/slot)', 'm-'),
        'spectral_eff': ('Spectral Efficiency', 'g-'),
        'collision_rate': ('Collision Rate', 'k-')
    }
    
    for metric, (title, style) in metrics_to_plot.items():
        plt.figure(figsize=(8, 5))
        plt.plot(metrics['episode'], metrics[metric], style)
        plt.xlabel('Episode')
        plt.ylabel(title)
        plt.title(title + ' vs Episodes')
        plt.grid(True)
        
        # Special y-axis limits for certain metrics
        if metric == 'urllc_success':
            plt.ylim(0.8, 1.05)
        elif metric == 'mmtc_throughput':
            plt.ylim(0.7, 1.2)
        elif metric == 'collision_rate':
            plt.ylim(0, max(0.02, max(metrics[metric]) * 1.1))
        
        plt.savefig(f'{output_dir}/plots/{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()