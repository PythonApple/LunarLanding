from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from GymLunarLander import GymLunarLander
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit


class CompetitiveEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.agents = []
        self.current_agent = 1
    

    def add_agent(self, agent):
        self.agents.append(agent)
    
    def reset(self, seed = None, options = None):
        obs = self.env.reset()
        return obs
    
    def swap_agent(self):
        self.current_agent = 1 - self.current_agent
    
    def step(self, action):
        # The environment needs to know which agent is taking the action
        current_obs, reward, done, truncated, info = self.env.step(action, self.current_agent)
       
        # Opponents move
        opponent = 1-self.current_agent
        opponent_obs, _, _, _, _ = self.env.step(0,opponent)
        opponent_action = self.agents[opponent].predict(opponent_obs)[0]
        self.env.step(opponent_action, opponent)
        
        return current_obs, reward, done, truncated, info

base_env = GymLunarLander(render_mode="rgb_array")
CompEnv = CompetitiveEnvWrapper(base_env)
timelimit_env = TimeLimit(CompEnv, max_episode_steps=1000)  # 1000 steps per episode

env = RecordVideo(
    timelimit_env,
    video_folder="./videos",
    episode_trigger=lambda x: x % 100 == 0,  # Record every 100 episodes
    disable_logger=True
)


agent1 = PPO("MlpPolicy", env, verbose=1)
agent2 = PPO("MlpPolicy", env, verbose=1)


CompEnv.add_agent(agent1)
CompEnv.add_agent(agent2)



# Training loop
for iteration in range(10000):

    agent1.learn(total_timesteps=10000, reset_num_timesteps=False)
    CompEnv.swap_agent()
    agent2.learn(total_timesteps=10000, reset_num_timesteps=False)
    CompEnv.swap_agent()

    print(f"Iteration {iteration + 1} complete.")
print("Training complete.")