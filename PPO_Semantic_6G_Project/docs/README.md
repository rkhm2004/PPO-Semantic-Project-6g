# PPO-Based Semantic-Aware Access for 6G Networks

## Overview
This project implements a PPO-based reinforcement learning approach for 6G network access, inspired by the paper "A Semantic-Aware Multiple Access Scheme for Distributed, Dynamic 6G-Based Applications" (arXiv:2401.06308). It optimizes URLLC latency and mMTC throughput with semantic reuse.

## Setup
1. Install Python 3.13.5 globally.
2. Install dependencies: `pip install gym stable-baselines3 numpy matplotlib`.
3. Run the simulation: `python src/main.py`.

## Novelty
Replaces SAMA-D3QL's DQN with PPO, introducing a stable policy-based method for semantic-aware 6G access.