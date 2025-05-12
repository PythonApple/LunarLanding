from stable_baselines3 import PPO
from GymLunarLander import GymLunarLander
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import TimeLimit
import random
from stable_baselines3.common.vec_env import SubprocVecEnv


class CompetitiveEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.teams = {}
        self.current_team = None
 
    def add_team(self, team_name, team):
        self.teams[team_name] = team
    
    def reset(self, seed = None, options = None):
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        team1_reward = 0
        team2_reward = 0

        if self.current_team == "RED":
            obs, reward, done, _, info = self.env.step(0, 0)
            #team1_reward += reward
            obs, reward, done, _, info = self.env.step(action[1], 1)
            team1_reward += reward
            #obs, reward, done, _, info = self.env.step(self.teams["BLUE"].predict(obs)[0][0], 2)
            obs, reward, done, _, info = self.env.step(0, 2)
            team2_reward += reward
            obs, reward, done, _, info = self.env.step(0, 3)
            team2_reward += reward

        if self.current_team == "BLUE":
            obs, reward, done, _, info = self.env.step(action[0], 2)
            team2_reward += reward
            obs, reward, done, _, info = self.env.step(action[1], 3)
            team2_reward += reward
            obs, reward, done, _, info = self.env.step(self.teams["RED"].predict(obs)[0][0], 0)
            team1_reward += reward
            obs, reward, done, _, info = self.env.step(self.teams["RED"].predict(obs)[0][1], 1)
            team1_reward += reward

        if self.current_team == "RED":
            return obs, team1_reward - 0, done, False, info
        elif self.current_team == "BLUE":
            return obs, team2_reward - team1_reward, done, False, info
  


base_env = GymLunarLander(render_mode="rgb_array")
CompEnv = CompetitiveEnvWrapper(base_env)
timelimit_env = TimeLimit(CompEnv, max_episode_steps=1000)  # 1000 steps per episode)
env = RecordVideo(timelimit_env, "videos", episode_trigger=lambda x: x % 100 == 0)

team1 = PPO.load("team1", env=env, verbose=2)
team2 = PPO.load("team2", env=env, verbose=2)

CompEnv.add_team("RED", team1)
CompEnv.add_team("BLUE", team2)

# Training loop
for iteration in range(100000):

    CompEnv.current_team = "RED"
    team1.learn(total_timesteps=15_000, reset_num_timesteps=False)
    #CompEnv.current_team = "BLUE"
    #team2.learn(total_timesteps=15_000, reset_num_timesteps=False)

    if iteration % 3 == 0:
        team1.save("team1")
        team2.save("team2")

    print(f"Iteration {iteration + 1} complete.")
print("Training complete.")

#tensorboard --logdir file:///Users/xingsun/LunarLanding/logs
#/Users/xingsun/Downloads/events.out.tfevents.1746838080.SL-2ZN2SL3.8560.0