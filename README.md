# AI-Driven Semantic-Aware Multiple Access (SAMA) for 6G Networks

This repository contains the implementation of a comprehensive resource allocation framework for 6G networks using Reinforcement Learning. The project explores both **Single-Agent PPO** and **Multi-Agent (CTDE/MAPPO-like)** approaches to optimize channel access for heterogeneous traffic types.

## 📌 Project Overview
The vision of 6G connectivity requires serving diverse applications with conflicting requirements. This project addresses dynamic resource allocation within a 6G network slice, balancing high-priority **URLLC** (Ultra-Reliable Low-Latency Communications) and best-effort **mMTC** (massive Machine-Type Communications).

### Key Features
* **Hybrid Dataset Pipeline**: Converts raw CSV traffic logs into structured, slot-based activation models with burst shaping.
* **Gymnasium-Style Environment**: A custom simulation environment designed with multi-objective reward shaping.
* **Sim-to-Real Methodology**: Models are trained on real traffic traces and fine-tuned using synthetic augmentations to ensure robustness against unpredictable network dynamics.
* **Multi-Agent Coordination**: Employs a Centralized Training, Decentralized Execution (CTDE) paradigm where independent agents learn cooperative scheduling policies.

---

## 🛠 Workflow Summary
1. **Preprocessing**: Raw 6G traffic logs are converted into fixed-horizon activation matrices.
2. **Environment Initialization**: Custom SAMA (Single-Agent) or SAMA-MADRL (Multi-Agent) environment setup following the Gymnasium API.
3. **Training**: Proximal Policy Optimization (PPO) agents learn to manage 3 channels across 300 time slots.
4. **Evaluation**: Capturing key metrics including URLLC success, mMTC throughput, and spectral efficiency.



---

## 📊 Performance Metrics
The system is evaluated based on the following criteria:
* **URLLC Success Rate**: Stringent reliability for mission-critical applications.
* **mMTC Throughput**: Optimizing data transfer for a vast number of low-priority devices.
* **Spectral Efficiency**: Total successful transmissions relative to available bandwidth.
* **Collision Rate**: Monitoring conflict events to maintain network stability.

### Experimental Results
* **URLLC Reliability**: The learned policies consistently achieved URLLC success rates of **0.98–1.00**.
* **Channel Utilization**: The SAMA-MADRL (Multi-Agent) model demonstrated balanced utilization across all channels (~32–34% each), reducing traffic bias compared to single-agent baselines.
* **Collision Minimization**: The multi-agent approach reduced collision rates to approximately **0.016 per slot**.

---

## 📂 Dataset Summary
The project uses three core structured files for simulation:
* `real_initial_states.csv`: Defines UE identity (6 UEs), traffic types, and priority levels.
* `real_traffic_model.csv`: Binary activation indicators over 300 time slots.
* `real_semantic_segments.csv`: Maps UEs to semantic labels (e.g., Quiz, VR-session, Discussion).

---

## 🚀 Key Technologies
* **Deep Learning**: PyTorch.
* **Reinforcement Learning**: Stable Baselines3 (PPO).
* **Environments**: Gymnasium & PettingZoo.
* **Data Processing**: NumPy & Pandas.

---

