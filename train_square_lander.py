import gymnasium as gym
from stable_baselines3 import PPO
from square_lunar_lander import SquareLunarLander
from gymnasium.wrappers import RecordVideo


# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")  # Disable rendering during training

env = RecordVideo(
    env,
    video_folder="./videos",  # Where to save videos
    episode_trigger=lambda x: x % 10 == 0,  # Record every 10 episodes
    disable_logger=True
)

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,          # Show training logs
    device="auto",      # Uses GPU if available
    learning_rate=1e-4,
    ent_coef=0.1,       # Encourage exploration
    clip_range=0.1,
    n_steps=1024,       # Steps per environment per update
    batch_size=32,      # Batch size
    gamma=0.99,         # Discount factor
)

# Train the model
model.learn(
    total_timesteps=100_000,  # Train for 500k steps (~250 episodes)
    progress_bar=True        # Shows nice progress bar
)

# Save the trained model
model.save("square_lander_basic")
print("Training complete! Model saved.")

env.close()

