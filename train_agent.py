import gymnasium as gym
from stable_baselines3 import PPO

# Create the MuJoCo environment
env = gym.make("HalfCheetah-v4", render_mode="human")
#this environment is good for testing continuous control algorithms.
#halfcheetah is a 2D robot that resembles a cheetah and is used for benchmarking reinforcement learning algorithms.
# Initialize the environment

# Define the PPO model
model = PPO("MlpPolicy", env, verbose=1)
#this model is good for mujoco environments.
# Train the model
print("--- Starting Training ---")
model.train(total_timesteps=1_000_000)
#change the timesteps to 5_000_000 for better results.
print("--- Training Finished ---")

# Test the trained agent
print("\n--- Displaying Trained Agent ---")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()