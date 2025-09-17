import gymnasium as gym
from stable_baselines3 import PPO
env= gym.make("HalfCheetah-v4", render_mode="human")
model=PPO("MlpPolicy", env,verbose=1)
print("--- Starting Training ---")
model.learn(total_timesteps=1_000_000)
print("--- Training Finished ---")
print("\n--- Displaying Trained Agent ---")
obs, info=env.reset()
for _ in range(1000):
	action, _states= model.predict(obs, deterministic=True)
	obs, reward, terminated, truncated, info= env.step(action)
	if terminated or truncated:
		obs, info= env.reset()
env.close()

