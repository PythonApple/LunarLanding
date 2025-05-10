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
        self.current_agent = None
 
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def reset(self, seed = None, options = None):
        obs = self.env.reset()
        return obs
    
    def step(self, action):

        team1_score = 0
        team2_score = 0

        obs, reward, done, truncated, info = self.env.step(action, self.current_agent)
        if self.current_agent in [0,1]:
            team1_score += reward
        elif self.current_agent in [2,3]:
            team2_score += reward

        obs, reward, done, truncated, info = self.env.step(0, 0)
        obs, reward, done, truncated, info = self.env.step(0, 2)
        obs, reward, done, truncated, info = self.env.step(0, 3)

        
        # Opponents move
#        if (self.current_agent != 0):
 #           obs, reward, done, truncated, info = self.env.step(self.agents[0].predict(obs)[0], 0)
 #           team1_score += reward
 #       if (self.current_agent != 1):
 #           obs, reward, done, truncated, info = self.env.step(self.agents[1].predict(obs)[0], 1)
 #           team1_score += reward
 #       if (self.current_agent != 2):
 #           obs, reward, done, truncated, info = self.env.step(self.agents[2].predict(obs)[0], 2)
 #           team2_score += reward
 #       if (self.current_agent != 3):
 #           obs, reward, done, truncated, info = self.env.step(self.agents[3].predict(obs)[0], 3)
 #           team2_score += reward
      

        if self.current_agent in [0,1]:
            return obs, team1_score - team2_score, done, truncated, info
        elif self.current_agent in [2,3]:
            return obs, team2_score - team1_score, done, truncated, info
            
  
 
def make_env(rank):
    base_env = GymLunarLander(render_mode="rgb_array")
    CompEnv = CompetitiveEnvWrapper(base_env)
    timelimit_env = TimeLimit(CompEnv, max_episode_steps=1000)  # 1000 steps per episode

    env = RecordVideo(
        timelimit_env,
        video_folder="./4Game",
        episode_trigger=lambda x: x % 100 == 0,  # Record every 100 episodes
        disable_logger=True
    )

    return env

env = SubprocVecEnv([make_env(i) for i in range(5)]) 


#agent1 = PPO("MlpPolicy", env=env, verbose=1)
#agent2 = PPO("MlpPolicy", env=env, verbose=1)
#agent3 = PPO("MlpPolicy", env=env, verbose=1)
#agent4 = PPO("MlpPolicy", env=env, verbose=1)

agent1 = PPO.load("Agent1_4Game", env=env, verbose=1)
agent2 = PPO.load("Agent2_4Game", env=env, verbose=1)
agent3 = PPO.load("Agent3_4Game", env=env, verbose=1)
agent4 = PPO.load("Agent4_4Game", env=env, verbose=1)



CompEnv.add_agent(agent1)
CompEnv.add_agent(agent2)
CompEnv.add_agent(agent3)
CompEnv.add_agent(agent4)



# Training loop
for iteration in range(1000000):

    CompEnv.current_agent = 0
    agent1.learn(total_timesteps=1, reset_num_timesteps=False)
    CompEnv.current_agent = 1
    agent2.learn(total_timesteps=1, reset_num_timesteps=False)
    CompEnv.current_agent = 2
    agent3.learn(total_timesteps=1, reset_num_timesteps=False)
    CompEnv.current_agent = 3
    agent4.learn(total_timesteps=1, reset_num_timesteps=False)


    if iteration % 10 == 0:
        agent1.save("Agent1_4Game")  # Overwrites previous save
        agent2.save("Agent2_4Game")  # One line per agent
        agent3.save("Agent3_4Game")
        agent4.save("Agent4_4Game")
        

    print(f"Iteration {iteration + 1} complete.")
print("Training complete.")

#tensorboard --logdir file:///Users/xingsun/LunarLanding/logs