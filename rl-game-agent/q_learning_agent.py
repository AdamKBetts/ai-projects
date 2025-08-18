import gymnasium as gym
import numpy as np
import random

# Create the environment
env = gym.make("Taxi-v3")
state_space = env.observation_space.n
action_space = env.action_space.n

# Initialize the Q-table
q_table = np.zeros((state_space, action_space))

# Set up the hyperparameters for the algorithm
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

num_episodes = 20000

# The Q-learning algorithm
for episode in range(num_episodes):
    state, info = env.reset()
    done = False

    # Exploration-exploitation trade-off
    exploration_rate_threshold = random.uniform(0, 1)
    if exploration_rate_threshold > exploration_rate:
        action = np.argmax(q_table[state, :])
    else:
        action = env.action_space.sample()
    
    while not done:
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update the Q-table using the Bellman equation
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state

        # Choose the next action
        if random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

    # Decay the exploration rate
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

print("Training complete!")

# Test the trained agent
print("Testing trained agent...")
test_env = gym.make("Taxi-v3", render_mode="human")
state, info = test_env.reset()
done = False
while not done:
    action = np.argmax(q_table[state, :])
    state, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated

test_env.close()

print("Agent test complete!")