# Project 6: Reinforcement Learning Game Agent

This project is an introduction to **Reinforcement Learning (RL)**, a field of AI where an agent learns to make optimal decisions by interacting with an environment. The agent, in this case, a **Q-Learning** algorithm, learns to navigate and solve the classic `Taxi-v3` environment from the Gymnasium library. This demonstrates the fundamental principles of training an AI through rewards and punishments.

---

### Key Concepts 🧠

- **Agent:** The AI entity that makes decisions.
- **Environment:** The virtual world the agent interacts with (`Taxi-v3`).
- **State:** The current condition of the environment (e.g., the taxi's location).
- **Action:** A move the agent can make (e.g., picking up or dropping off a passenger).
- **Reward:** A numerical value the agent receives for a good action, which guides its learning.

---

### Features ✨

- **Self-Learning Agent:** The agent learns the optimal policy through thousands of interactions with the environment.
- **Q-Learning Algorithm:** Implements a foundational RL algorithm for training the agent.
- **Open-Source Environments:** Uses the popular `gymnasium` library, which provides a standard interface for RL environments.
- **Human-in-the-Loop Demonstration:** After training, the agent plays the game in a visible window, showcasing what it has learned.

---

### Installation 🛠️

1.  **Navigate to the project directory:**

    ```bash
    cd rl-game-agent
    ```

2.  **Install the required libraries:**
    ```bash
    pip install gymnasium numpy
    ```

---

### Usage ▶️

1.  **Run the script:**

    ```bash
    python q_learning_agent.py
    ```

2.  **Observe the training:**
    The script will first perform 20,000 training episodes in the background. This process takes a few moments.

3.  **Watch the agent play:**
    Once training is complete, a new window will pop up showing the agent autonomously playing the game, which demonstrates its learned behavior.

---
