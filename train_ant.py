import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the four-legged 'Ant' environment
env = gym.make("Ant-v4", render_mode="human")

# 2. Define the PPO model with tuned hyperparameters
# These settings help the agent learn more carefully on a complex task.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=4096,          # Collect more data before each update
    learning_rate=0.0001   # Learn more slowly and carefully
)

# 3. Train the model for an extended period
print("--- Starting Ant Training ---")
model.learn(
    total_timesteps=10_000_000 # Increased training time for the harder environment
)
print("--- Training Finished ---")

# Save the trained model
model.save("ppo_ant_model")


# 4. Test the trained agent
print("\n--- Displaying Trained Ant ---")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        
env.close()
#