import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# A simple Grid World environment with obstacles and stochasticity
class GridWorld:
    def __init__(self, size=5, obstacles=None, stochasticity=0.1):
        self.size = size
        self.stochasticity = stochasticity
        self.obstacles = obstacles if obstacles is not None else []
        
        self.grid = np.zeros((size, size))
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = self.start_pos
        
        self.grid[self.goal_pos] = 1 # Mark the goal
        for obs in self.obstacles:
            self.grid[obs] = -1 # Mark obstacles
            if obs == self.start_pos or obs == self.goal_pos:
                raise ValueError(f"Obstacle at {obs} cannot be at start or goal.")

        self.actions = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.action_space = len(self.actions)
        self.state_space = size * size

    def reset(self):
        self.agent_pos = self.start_pos
        state_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        return state_idx

    def step(self, action):
        # Stochastic environment: action may not be the one taken
        if np.random.rand() < self.stochasticity:
            action = np.random.randint(0, self.action_space)

        y, x = self.agent_pos
        if action == 0: # UP
            y = max(0, y - 1)
        elif action == 1: # DOWN
            y = min(self.size - 1, y + 1)
        elif action == 2: # LEFT
            x = max(0, x - 1)
        elif action == 3: # RIGHT
            x = min(self.size - 1, x + 1)

        intended_pos = (y, x)
        reward = -0.1 # Small penalty for each step
        done = False

        # Check for collisions with obstacles
        if intended_pos in self.obstacles:
            new_pos = self.agent_pos # Stay in place
            reward = -1.0 # Increased penalty for hitting a wall
        else:
            new_pos = intended_pos

        if new_pos == self.goal_pos:
            reward = 1.0 # Large reward for reaching the goal
            done = True
        
        self.agent_pos = new_pos
        state_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        return state_idx, reward, done, {}

# --- ACTOR AND CRITIC WITH INCREASED NETWORK CAPACITY ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state):
        return self.network(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.data_buffer = []
        
    def select_action(self, state):
        state_one_hot = torch.zeros(env.state_space)
        state_one_hot[state] = 1.0
        state_tensor = state_one_hot.float()
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self):
        old_log_probs = torch.tensor([lp for _, _, _, lp in self.data_buffer], dtype=torch.float)
        actions = torch.tensor([a for _, a, _, _ in self.data_buffer], dtype=torch.long)
        rewards = [r for _, _, r, _ in self.data_buffer]
        states_one_hot = torch.zeros(len(self.data_buffer), env.state_space)
        for i, s_idx in enumerate([s for s, _, _, _ in self.data_buffer]):
            states_one_hot[i][s_idx] = 1.0
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        with torch.no_grad():
            V_old = self.critic(states_one_hot).squeeze()
        advantages = discounted_rewards - V_old
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # --- UPDATE LOOP WITH INCREASED ENTROPY BONUS ---
        for _ in range(self.K_epochs):
            action_probs = self.actor(states_one_hot)
            dist = Categorical(logits=action_probs)
            new_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy().mean()
            V_new = self.critic(states_one_hot).squeeze()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = nn.functional.mse_loss(V_new, discounted_rewards)
            
            # Increased entropy coefficient to encourage more exploration
            total_loss = actor_loss + 0.5 * critic_loss - 0.02 * dist_entropy
            
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            total_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()
        self.data_buffer.clear()

def visualize_policy(agent, env):
    print("\n--- Learned Policy ---")
    policy_grid = [['' for _ in range(env.size)] for _ in range(env.size)]
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for y in range(env.size):
        for x in range(env.size):
            if (y, x) == env.goal_pos:
                policy_grid[y][x] = 'G'
                continue
            if (y, x) in env.obstacles:
                policy_grid[y][x] = 'X'
                continue
            state_idx = y * env.size + x
            state_one_hot = torch.zeros(env.state_space)
            state_one_hot[state_idx] = 1.0
            with torch.no_grad():
                action_logits = agent.actor(state_one_hot)
                best_action = torch.argmax(action_logits).item()
                policy_grid[y][x] = action_symbols[best_action]
    for row in policy_grid:
        print(' | '.join(f'{cell:^3}' for cell in row))
    print("G: Goal, X: Obstacle, Arrows: Agent's most likely action\n")


if __name__ == '__main__':
    # Hyperparameters
    state_dim = 25
    action_dim = 4
    lr_actor = 0.0001
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2

    # Setup with obstacles
    obstacles = [(1, 1), (1, 2), (1, 3), (3, 3), (3, 4), (4, 1)]
    env = GridWorld(size=5, obstacles=obstacles, stochasticity=0.1)
    agent = PPOAgent(env.state_space, env.action_space, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    
    all_rewards = []
    
    # Training loop
    num_episodes = 5000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.data_buffer.append((state, action, reward, log_prob.item()))
            state = next_state
            total_reward += reward
        agent.update()
        all_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Average Reward (last 100): {np.mean(all_rewards[-100:]):.2f}")

    # Plotting and visualization
    plt.figure(figsize=(12, 6))
    plt.plot(all_rewards)
    moving_avg = np.convolve(all_rewards, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(99, len(all_rewards)), moving_avg, color='red', linewidth=2, label='100-episode Moving Average')
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig('reward_plot.png')
    print("\nReward plot saved as 'reward_plot.png'")
    plt.show()

    visualize_policy(agent, env)