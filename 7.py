import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create Environment
# -------------------------------
env = gym.make("CartPole-v1")

n_actions = env.action_space.n
n_states = 20
episodes = 300

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

# -------------------------------
# Step 2: Discretization Helpers
# -------------------------------
def create_bins(n_states, env):
    bins = []
    for i in range(env.observation_space.shape[0]):
        low, high = env.observation_space.low[i], env.observation_space.high[i]

        # Handle infinite bounds
        if low == -np.inf:
            low = -4
        if high == np.inf:
            high = 4

        bins.append(np.linspace(low, high, n_states - 1))
    return bins

def discretize_state(state, bins):
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))

state_bins = create_bins(n_states, env)

# Q-table
q_table = np.zeros((n_states, n_states, n_states, n_states, n_actions))

# -------------------------------
# Step 3: Training Loop
# -------------------------------
rewards = []

for episode in range(episodes):
    state_raw, info = env.reset()
    state = discretize_state(state_raw, state_bins)
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state_raw, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state_raw, state_bins)

        # Q-learning update
        q_table[state][action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

    print(f"Episode {episode+1}/{episodes} | Reward: {total_reward} | Epsilon: {epsilon:.3f}")

env.close()

# -------------------------------
# Step 4: Plot Rewards
# -------------------------------
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Progress")
plt.show()

# -------------------------------
# Step 5: Save Q-table
# -------------------------------
np.save("q_table.npy", q_table)
print("Q-table saved as q_table.npy")

# -------------------------------
# Step 6: Test Trained Agent
# -------------------------------
env = gym.make("CartPole-v1", render_mode="human")

state_raw, info = env.reset()
state = discretize_state(state_raw, state_bins)
done = False
test_reward = 0

while not done:
    action = np.argmax(q_table[state])
    next_state_raw, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = discretize_state(next_state_raw, state_bins)
    test_reward += reward

env.close()
print("Total reward during test:", test_reward)

