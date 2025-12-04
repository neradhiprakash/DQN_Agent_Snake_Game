# Snake Game – Deep Q-Network (DQN) Agent

This repository contains my **Deep Q-Network (DQN)** implementation for the classic Snake game.  
The project explores reinforcement learning techniques — including epsilon-greedy exploration, experience replay, target networks, and reward shaping — to train an autonomous agent capable of playing Snake.

>  **Note:**  
> This repository includes **only my personal contribution** to a larger group project.  
> Other algorithms (BFS, DFS, A*) were implemented by teammates and are **not included** here to respect authorship and academic integrity.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Group Project Disclosure](#group-project-disclosure)
- [Citation](#citation)

---

## Project Overview

This project implements a Deep Q-Network to train an agent to play the Snake game using reinforcement learning.  
The model uses:

- Neural network–based Q-value approximation  
- Experience replay buffer  
- Target network updates  
- Epsilon-greedy exploration  
- Custom reward shaping  
- Training visualizations (scores, epsilon decay, exploration rate)

The final trained model is included (`best_model.pth`), along with scripts to replay the agent.

---

## Features

- 1. Fully functional Snake game environment  
- 2. DQN agent with replay memory  
- 3. Training loop and checkpoint saving  
- 4. Trained model evaluation  
- 5. Score, epsilon, and exploration plots  
- 6. Clean modular code (`src/` folder)

---

## Repository Structure

```text
Snake_DQN_Agent/
│
├── models/
│   └── best_model.pth
│
├── data/
│   └── training_log.csv
│
├── src/
│   ├── dqn_agent.py
│   ├── train_dqn.py
│   ├── play_trained_snake.py
│   ├── snake_game.py
│   └── plot_training.py
│
├── plots/
│   ├── 1_scores_plot.png
│   ├── 2_epsilon_plot.png
│   └── 3_exploration_plot.png
│
├── report/
│   └── CSD311_FinalProjectReport_Group17.pdf
│
├── .gitignore
└── README.md
```

## How to Run

### 1. Install dependencies
This project uses PyTorch, NumPy, Pygame, and Matplotlib.

```bash
pip install torch numpy pygame matplotlib
```

### 2. Train the DQN agent
Runs the full training loop, saves the best model, and logs training stats.
```bash
python src/train_dqn.py
```
### 3. Watch the trained agent play Snake
Loads best_model.pth and plays automatically.
```bash
python src/play_trained_snake.py
```

### 4. Regenerate training plots
This script rebuilds score, epsilon, and exploration plots.
```bash
python src/plot_training.py
```

## Disclosure
This repository contains only my personal contribution to a larger group project
submitted for CSD311 – Artificial Intelligence (Monsoon 2025).

### Included (my work):

- DQN agent implementation
- Training loop + replay buffer
- Neural network architecture
- Model saving, inference, and evaluation
- Visualizations (scores, epsilon decay, exploration)
- My DQN analysis
- Methodology and Findings of the project

### Not included (authored by teammates):

- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- A* Search
These components are omitted to respect authorship and academic integrity.

### Citations
If you use or reference this work, please cite both:
1. Prakash, N. (2025). *Snake Game – Deep Q-Network (DQN) Agent*. 
GitHub Repository: https://github.com/neradhiprakash/DQN_Agent_Snake_Game

2. Group 17 (2025). *CSD311 – Artificial Intelligence: Final Project Report*. 
Shiv Nadar University, Monsoon 2025.

## Contact
For questions, feedback, or collaboration:
aneradhiprakash@gmail.com

