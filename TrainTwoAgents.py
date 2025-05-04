from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from lunar_landar_twoAgents import SquareLunarLanderTwo
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


class CompetitiveEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.current_agent = None  # Track which agent is currently acting
    
    def reset(self, seed = None, options = None):
        obs = self.env.reset()
        self.current_agent = 0  # Start with agent 0
        return obs
    
    def step(self, action):
        # The environment needs to know which agent is taking the action
        obs, reward, done, truncated, info = self.env.step(action, self.current_agent)
        
        # Switch agents for next step
        self.current_agent = 1 if self.current_agent == 0 else 0
        
        return obs, reward, done, truncated, info

# Original environment
base_env = SquareLunarLanderTwo(render_mode="rgb_array")
# Wrapped environment
env = CompetitiveEnvWrapper(base_env)

env = RecordVideo(
    env,
    video_folder="./videos",  # Where to save videos
    episode_trigger=lambda x: x % 1000 == 0,  # Record every 100 episodes
    disable_logger=True
)

agent1 = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))
agent2 = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

# Training loop
for iteration in range(10000):
    # Train Agent 0 (freeze Agent 1)
    env.current_agent = 0
    agent1.learn(total_timesteps=1, reset_num_timesteps=False)
    
    # Train Agent 1 (freeze Agent 0)
    env.current_agent = 1
    agent2.learn(total_timesteps=1, reset_num_timesteps=False)

    print(f"Iteration {iteration + 1} completed.")