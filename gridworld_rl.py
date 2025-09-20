import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

class GridWorldEnv:
    """GridWorld Environment for RL Navigation"""
    
    def __init__(self, size=8, obstacles=None):
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        
        # Default obstacles
        if obstacles is None:
            self.obstacles = [(2, 2), (2, 3), (3, 2), (5, 5), (5, 6), (6, 5)]
        else:
            self.obstacles = obstacles
            
        self.reset()
        
    def reset(self):
        """Reset environment to start position"""
        self.agent_pos = list(self.start_pos)
        return self.get_state()
    
    def get_state(self):
        """Get current state representation"""
        # Normalize position to [0, 1]
        state = np.array([
            self.agent_pos[0] / self.size,
            self.agent_pos[1] / self.size,
            (self.goal_pos[0] - self.agent_pos[0]) / self.size,  # Distance to goal
            (self.goal_pos[1] - self.agent_pos[1]) / self.size
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        # Convert continuous action to discrete for DDPG
        if isinstance(action, (list, np.ndarray)) and len(action) == 2:  # DDPG case
            # Map continuous actions to discrete moves
            if abs(action[0]) > abs(action[1]):
                discrete_action = 2 if action[0] < 0 else 3  # LEFT or RIGHT
            else:
                discrete_action = 0 if action[1] < 0 else 1  # UP or DOWN
        else:  # PPO case (action is an integer)
            discrete_action = action
        
        # Action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        old_pos = tuple(self.agent_pos)
        new_pos = [
            self.agent_pos[0] + moves[discrete_action][0],
            self.agent_pos[1] + moves[discrete_action][1]
        ]
        
        # Check boundaries and obstacles
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and 
            tuple(new_pos) not in self.obstacles):
            self.agent_pos = new_pos
        
        # Calculate reward
        reward = self.get_reward(old_pos, tuple(self.agent_pos))
        done = tuple(self.agent_pos) == self.goal_pos
        
        return self.get_state(), reward, done
    
    def get_reward(self, old_pos, new_pos):
        """Calculate reward for the transition"""
        # Goal reached
        if new_pos == self.goal_pos:
            return 100.0
        
        # Hit wall or obstacle (didn't move)
        if old_pos == new_pos:
            return -10.0
        
        # Distance-based reward
        old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
        new_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
        
        if new_dist < old_dist:
            return 1.0  # Getting closer
        else:
            return -0.1  # Getting farther or staying same distance


class PPONetwork(nn.Module):
    """PPO Actor-Critic Network"""
    
    def __init__(self, state_dim=4, action_dim=4, hidden_dim=64):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value)
        self.critic_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        # Shared representation
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))
        
        # Actor output (action probabilities)
        action_logits = self.actor_fc(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic output (state value)
        value = self.critic_fc(x)
        
        return action_probs, value


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_dim=4, action_dim=4, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.epochs = 10
        self.gae_lambda = 0.95
        
        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def get_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.network(state)
            
        # Sample action from probability distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition for later training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """Update policy using PPO algorithm"""
        if len(self.states) == 0:
            return
            
        # Convert to tensors properly
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones, dtype=np.float32)).to(self.device)
        
        # Calculate advantages using GAE
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        for _ in range(self.epochs):
            # Get current policy outputs
            action_probs, current_values = self.network(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear storage
        self.clear_storage()
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[i])
            advantages[i] = gae
            
        return advantages
    
    def clear_storage(self):
        """Clear trajectory storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


class DDPGActor(nn.Module):
    """DDPG Actor Network"""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Output in [-1, 1]
        return action


class DDPGCritic(nn.Module):
    """DDPG Critic Network"""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ReplayBuffer:
    """Experience Replay Buffer for DDPG"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent"""
    
    def __init__(self, state_dim=4, action_dim=2, lr_actor=1e-3, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = DDPGActor(state_dim, action_dim).to(self.device)
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Soft update parameter
        self.noise_scale = 0.1
        self.batch_size = 64
        
    def get_action(self, state, add_noise=True):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, action.shape)
            action = np.clip(action + noise, -1, 1)
            
        return action
    
    def update(self):
        """Update networks using DDPG algorithm"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (self.gamma * target_q * ~dones.unsqueeze(1))
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
    
    def soft_update(self, target, source):
        """Soft update target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


def train_agent(agent_type='PPO', episodes=1000, max_steps=100):
    """Train agent on GridWorld environment"""
    
    env = GridWorldEnv()
    
    if agent_type == 'PPO':
        agent = PPOAgent()
    else:  # DDPG
        agent = DDPGAgent()
    
    scores = []
    episode_lengths = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # PPO trajectory collection
        if agent_type == 'PPO':
            for step in range(max_steps):
                action, log_prob, value = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                agent.store_transition(state, action, reward, log_prob, value, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Update after episode
            agent.update()
            
        else:  # DDPG training
            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                agent.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update networks
                if len(agent.replay_buffer) > agent.batch_size:
                    agent.update()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
        
        scores.append(total_reward)
        episode_lengths.append(steps)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"{agent_type} Episode {episode}: Avg Score: {avg_score:.2f}, Avg Length: {avg_length:.1f}")
    
    return agent, scores, episode_lengths


def visualize_policy(agent, agent_type='PPO', env=None):
    """Visualize learned policy on the grid"""
    if env is None:
        env = GridWorldEnv()
    
    policy_grid = np.zeros((env.size, env.size, 2))  # Store best action for each position
    
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) not in env.obstacles:
                env.agent_pos = [i, j]
                state = env.get_state()
                
                if agent_type == 'PPO':
                    action_probs, _ = agent.network(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.argmax(action_probs, dim=1).item()
                    # Convert discrete action to arrow components
                    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
                    policy_grid[i, j] = moves[action]
                else:  # DDPG
                    action = agent.get_action(state, add_noise=False)
                    policy_grid[i, j] = action
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create grid visualization
    grid_vis = np.ones((env.size, env.size))
    for obs in env.obstacles:
        grid_vis[obs] = 0
    
    # Plot grid
    ax.imshow(grid_vis, cmap='RdYlGn', alpha=0.3)
    
    # Plot policy arrows
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) not in env.obstacles:
                dx, dy = policy_grid[i, j]
                ax.arrow(j, i, dy*0.3, dx*0.3, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Mark start and goal
    ax.scatter(env.start_pos[1], env.start_pos[0], c='green', s=200, marker='s', label='Start')
    ax.scatter(env.goal_pos[1], env.goal_pos[0], c='red', s=200, marker='*', label='Goal')
    
    # Mark obstacles
    for obs in env.obstacles:
        ax.scatter(obs[1], obs[0], c='black', s=200, marker='X')
    
    ax.set_title(f'{agent_type} Learned Policy')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_algorithms():
    """Compare PPO and DDPG performance"""
    
    print("Training PPO Agent...")
    ppo_agent, ppo_scores, ppo_lengths = train_agent('PPO', episodes=500)
    
    print("\nTraining DDPG Agent...")
    ddpg_agent, ddpg_scores, ddpg_lengths = train_agent('DDPG', episodes=500)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Smooth scores for better visualization
    window = 50
    ppo_smooth = np.convolve(ppo_scores, np.ones(window)/window, mode='valid')
    ddpg_smooth = np.convolve(ddpg_scores, np.ones(window)/window, mode='valid')
    
    ax1.plot(ppo_smooth, label='PPO', alpha=0.8)
    ax1.plot(ddpg_smooth, label='DDPG', alpha=0.8)
    ax1.set_title('Training Scores (Smoothed)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ppo_len_smooth = np.convolve(ppo_lengths, np.ones(window)/window, mode='valid')
    ddpg_len_smooth = np.convolve(ddpg_lengths, np.ones(window)/window, mode='valid')
    
    ax2.plot(ppo_len_smooth, label='PPO', alpha=0.8)
    ax2.plot(ddpg_len_smooth, label='DDPG', alpha=0.8)
    ax2.set_title('Episode Lengths (Smoothed)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Goal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize policies
    visualize_policy(ppo_agent, 'PPO')
    visualize_policy(ddpg_agent, 'DDPG')
    
    return ppo_agent, ddpg_agent


if __name__ == "__main__":
    # Run comparison
    print("GridWorld Navigation: PPO vs DDPG")
    print("=" * 40)
    
    ppo_agent, ddpg_agent = compare_algorithms()
    
    print("\nTraining completed! Check the plots for performance comparison.")
    print("\nKey Differences:")
    print("PPO: Discrete actions (UP/DOWN/LEFT/RIGHT), on-policy learning")
    print("DDPG: Continuous actions (direction vector), off-policy learning")
    print("PPO: Better sample efficiency, more stable")
    print("DDPG: Better for continuous control, more complex")