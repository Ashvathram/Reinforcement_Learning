# PPO Agent for MuJoCo's HalfCheetah 🤖

This project documents the process of training a Reinforcement Learning agent to control a "HalfCheetah" robot in the MuJoCo physics simulator. The agent is trained from scratch using the Proximal Policy Optimization (PPO) algorithm, implemented with the Stable-Baselines3 library.

The primary goal was not just to train an agent, but to build the entire development and training environment from the ground up, navigating numerous real-world setup and troubleshooting challenges along the way.


---
## Tech Stack
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Libraries:** PyTorch, Stable-Baselines3, Gymnasium
- **Simulation:** MuJoCo
- **OS & Environment:** Ubuntu 24.04 LTS running in a VirtualBox VM

---
## Setup and Installation
To replicate this environment on a fresh Ubuntu Desktop system:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install System Dependencies:**
    ```bash
    sudo apt update
    sudo apt install python3-venv git
    ```
3.  **Create and Activate a Python Virtual Environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
4.  **Install Python Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
---
## Usage
With the environment activated, start the training process by running:
```bash
python train_agent.py


The Journey & Troubleshooting Log 📜
This project was a significant learning experience that went far beyond the algorithm itself. This log documents the major hurdles encountered and their solutions.

Error 1: ModuleNotFoundError on Windows
Problem: Initial attempts to set up the environment on Windows failed during pip install with a ModuleNotFoundError: No module named 'distutils.msvccompiler'.

Root Cause: Python 3.12+ has removed the distutils library, which many packages still rely on for compiling C++ extensions on Windows.

Solution: The project was moved to a dedicated Ubuntu Virtual Machine, creating a clean and consistent Linux environment.

Error 2: Killed during pip install in the VM
Problem: The pip install process for PyTorch and its dependencies was abruptly terminated by the operating system.

Root Cause: This was an Out-of-Memory (OOM) error. The VM was only allocated 2GB of RAM, which was insufficient for the memory-intensive installation.

Solution: The VM was shut down, and its Base Memory (RAM) was increased to 4GB in the VirtualBox settings.

Error 3: python3-venv is not available
Problem: The command python3 -m venv env failed, stating that the tools to create virtual environments were missing.

Root Cause: The base Ubuntu installation did not include the python3-venv package by default.

Solution: The package was installed manually using sudo apt install python3-venv. This also required fixing a typo (pyton3) during the first installation attempt.

Error 4: TypeError in model.train()
Problem: The training script crashed with TypeError: PPO.train() got an unexpected keyword argument 'total_timesteps'.

Root Cause: An incorrect method was called. .train() is a lower-level internal method in Stable-Baselines3.

Solution: Corrected the method call from model.train() to model.learn(), which is the correct high-level API for running a full training session.

Future Work
1).Train the agent for more timesteps to achieve better performance.
2).Experiment with other MuJoCo environments (e.g., Hopper-v4, Walker2d-v4).
3).Tune the hyperparameters of the PPO algorithm (e.g., learning_rate, gamma) to observe changes in learning speed and final performance.
