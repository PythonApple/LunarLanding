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
    learning_rate=1e-4,       # Slower for stability
    n_steps=4096,            # More steps for complex strategies
    clip_range=0.1,          # Tighter clipping for competition
    ent_coef=0.02,           # Boost exploration
    verbose=1
)

# Train the model
model.learn(
    total_timesteps=500_000,  # Train for 500k steps (~250 episodes)
    progress_bar=True        # Shows nice progress bar
)

# Save the trained model
model.save("square_lander_basic")
print("Training complete! Model saved.")

env.close()

