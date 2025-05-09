from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from GymLunarLander import GymLunarLander
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit
import random
from stable_baselines3.common.vec_env import SubprocVecEnv



class CompetitiveEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.agents = []
        self.current_agent = 0
 
    

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
        _, opp_reward, _, _, _ = self.env.step(self.agents[opponent].predict(current_obs)[0], opponent)
        #_, opp_reward, _, _, _ = self.env.step(0, opponent)

        if (self.current_agent ==0):
            return current_obs, reward, done, truncated, info
        else:
            return current_obs, reward - opp_reward, done, truncated, info
  
 

base_env = GymLunarLander(render_mode="rgb_array")
CompEnv = CompetitiveEnvWrapper(base_env)
timelimit_env = TimeLimit(CompEnv, max_episode_steps=1000)  # 1000 steps per episode

env = RecordVideo(
    timelimit_env,
    video_folder="./videosRocket3",
    episode_trigger=lambda x: x % 1000 == 0,  # Record every 100 episodes
    disable_logger=True
)


agent1 = PPO.load("RocketDUOagent1Solo_latest", env=env, verbose=1,tensorboard_log=None)
agent2 = PPO.load("RocketDUOagent2Solo_latest", env=env, verbose=1, tensorboard_log=None)


CompEnv.add_agent(agent1)
CompEnv.add_agent(agent2)



# Training loop
for iteration in range(1000000):

    CompEnv.current_agent = 0
    agent1.learn(total_timesteps=1, reset_num_timesteps=False)
    CompEnv.current_agent = 1
    agent2.learn(total_timesteps=1, reset_num_timesteps=False)


    if iteration % 10 == 0:
        agent1.save("RocketDUOagent1Solo_latest")  # Overwrites previous save
        agent2.save("RocketDUOagent2Solo_latest")  # One line per agent

    print(f"Iteration {iteration + 1} complete.")
print("Training complete.")

#tensorboard --logdir file:///Users/xingsun/LunarLanding/logs