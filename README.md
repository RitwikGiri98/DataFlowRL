# 🚀 DataFlowRL  
### Reinforcement Learning for Data Pipeline Optimization  
**MS in Information Systems – Agentic AI Systems Final Project**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![RL](https://img.shields.io/badge/Reinforcement%20Learning-DQN%20%2B%20UCB-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🧠 Overview

**DataFlowRL** is an agentic AI system that uses **Reinforcement Learning (RL)** to optimize data pipeline execution.  
Instead of manually picking tools (e.g., Pandas, Spark, DuckDB) and configurations (low/balanced/high resources), an RL agent learns—through experience—which combinations minimize runtime and cost while maintaining data quality.

This project integrates:

- **DQN (Deep Q-Network)** for value-based policy learning  
- **UCB Bandit Explorer** for structured exploration  
- **Agentic orchestration** (Controller Agent, Planner Agent, Execution Agent)  
- **Custom Pipeline Simulator**  
- **Multi-dataset, multi-pipeline experiments**  
- **Full analysis, plots, and results**

The system is designed for professional-grade evaluation and applies RL research concepts to a real-world data engineering scenario.

---

## 🌟 Key Features

### ✔ Reinforcement Learning  
- Deep Q-Network (PyTorch)  
- Experience Replay + Target Network  
- Hybrid exploration: DQN ε-greedy + UCB Bandit  

### ✔ Agentic Orchestration  
- Controller Agent coordinates the entire episode loop  
- Specialized agents for planning, execution, and logging  
- Custom tool: `PipelineRunSimulator`

### ✔ Simulated Data Environment  
Simulates:
- Runtime  
- Cost  
- Quality  
- Failure probability  
for each (tool, config, step type, complexity, dataset size) combination.

### ✔ Multi-Scenario Experiments  
- 3 dataset scales (10k, 100k, 1M rows)  
- 3 pipeline structures (A, B, C)  
- 40 RL training episodes, 20 eval, 20 baseline episodes  
- Automatic CSV generation + visualizations

---

## 🏗 System Architecture

```mermaid
flowchart LR

    %% LAYER 1: USER
    subgraph L1["User / Experiment Setup"]
        U[User or Experiment Runner]
    end

    %% LAYER 2: CONTROLLER & AGENTS
    subgraph L2["Controller and Orchestration Layer"]
        C[Controller Agent]
        P[Pipeline Planner Agent]
        E[Execution Agent]
        M[Metrics Agent]
    end

    %% LAYER 3: RL MODULE
    subgraph L3["Reinforcement Learning Module"]
        Q[DQN Agent]
        RB[Replay Buffer]
        TQ[Target Q-Network]
        UCB[UCB Bandit Explorer]
    end

    %% LAYER 4: ENVIRONMENT
    subgraph L4["Pipeline Environment"]
        SE[State Encoder]
        RC[Reward Calculator]
    end

    %% LAYER 5: TOOLS AND DATA
    subgraph L5["Tools and Data Layer"]
        SIM[PipelineRunSimulator]
        DS[(Synthetic Datasets)]
        LOGS[(CSV Logs)]
    end

    %% FLOWS
    U --> |Chooses dataset and pipeline| C

    C --> |Build steps| P
    P --> |Pipeline steps| SE

    SE --> |State s_t| C
    C --> |Query action a_t| Q
    Q --> |Selected action| C

    C --> |Execute step| E
    E --> |Run step| SIM
    SIM --> |runtime, cost, quality, success| RC
    RC --> |Reward and next state| SE

    SE --> |Transition| Q
    Q --> |Store experience| RB
    Q --> |Periodic target update| TQ

    RC --> |Reward samples| UCB
    UCB --> |Exploration suggestion| C

    RC --> |Metrics| M
    M --> |Episode logs| LOGS

    DS --> SIM
