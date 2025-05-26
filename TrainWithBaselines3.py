from stable_baselines3 import PPO
from GymLunarLander import GymLunarLander
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.logger import TensorBoardOutputFormat
import os
import csv


class CompetitiveEnvWrapper(gym.Wrapper):
    def __init__(self, env, id):
        super().__init__(env)
        self.env = env
        self.agents = []
        self.agents.append(PPO.load("Agent1_4Game"))
        self.agents.append(PPO.load("Agent2_4Game"))
        self.agents.append(PPO.load("Agent3_4Game"))
        self.agents.append(PPO.load("Agent4_4Game"))
        self.current_agent = None
        self.episodes = 0
        self.total_timesteps = 0
        self.id = id
        self.log_path = "./logs/manual_log.csv"
        if self.id == 0:
            os.makedirs("./logs", exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "agent", "attribute", "value"])

    def reload_agents(self):
        self.agents = [PPO.load("Agent1_4Game"), PPO.load("Agent2_4Game"), PPO.load("Agent3_4Game"), PPO.load("Agent4_4Game")]

    def set_current(self, agent_id):
        self.current_agent = agent_id
 
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def reset(self, seed = None, options = None):
        self.episodes += 1
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        self.total_timesteps += 1

        team1_score = 0
        team2_score = 0

        obs, reward, done, truncated, info = self.env.step(action, self.current_agent)
        if self.current_agent in [0,1]:
            team1_score += reward
        elif self.current_agent in [2,3]:
            team2_score += reward 
        
        # Opponents move
        if (self.current_agent != 0):
            obs, reward, done, truncated, info = self.env.step(self.agents[0].predict(obs)[0], 0)
            team1_score += reward
        if (self.current_agent != 1):
            obs, reward, done, truncated, info = self.env.step(self.agents[1].predict(obs)[0], 1)
            team1_score += reward
        if (self.current_agent != 2):
            obs, reward, done, truncated, info = self.env.step(self.agents[2].predict(obs)[0], 2)
            team2_score += reward
        if (self.current_agent != 3):
            obs, reward, done, truncated, info = self.env.step(self.agents[3].predict(obs)[0], 3)
            team2_score += reward

        team1_score /= 5
        team2_score /= 5
   
        self.env.set_score(team2_score-team1_score)

        if self.id == 0 and self.episodes % 50 == 0:
            agent_names = ["r1", "r2", "b1", "b2"]
            attributes = [
                "px", "py", "sx", "sy",
                "a_", "a_s", "lC", "rC"
            ]
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                for i, agent in enumerate(agent_names):
                    for j, attr in enumerate(attributes):
                        index = i * 8 + j
                        writer.writerow([self.total_timesteps, agent, attr, f"{obs[index]:.3f}"])

        if self.current_agent in [0,1]:
            return obs, team1_score - team2_score, done, truncated, info
        elif self.current_agent in [2,3]:
            return obs, team2_score - team1_score, done, truncated, info

class ActionHistogramCallback(BaseCallback):
    def __init__(self, agent_id, log_interval=100, verbose=0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.log_interval = log_interval
        self.actions = []
        self.last_logged_step = 0

    def _on_step(self) -> bool:
        action = self.locals.get("actions", None)
        if action is not None:
            # Convert and store as integers
            if isinstance(action, np.ndarray):
                self.actions.extend(action.flatten().astype(int).tolist())
            elif isinstance(action, list):
                self.actions.extend([int(a) for a in action])
            else:
                self.actions.append(int(action))

        # Log every `log_interval` steps
        if (self.num_timesteps - self.last_logged_step) >= self.log_interval:
            self._log_histogram()
            self.last_logged_step = self.num_timesteps
            self.actions.clear()

        return True

    def _log_histogram(self):
        if not self.actions:
            return

        writer = None
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                writer = fmt.writer
                break

        if writer:
            actions_array = np.array(self.actions)
            writer.add_histogram(
                tag=f"agent{self.agent_id}/actions_histogram",
                values=actions_array,
                global_step=self.num_timesteps
            )

def make_env(id):
    def _init():
        base_env = GymLunarLander(render_mode="rgb_array")
        CompEnv = CompetitiveEnvWrapper(base_env, id)
        env = TimeLimit(CompEnv, max_episode_steps=500)  # 1000 steps per episode
        if id==0:
            env = RecordVideo(
                env,
                video_folder="./4Game",
                episode_trigger=lambda x: x % 50 == 0,  # Record every 100 episodes
                disable_logger=True
            )
        return env
    return _init

if __name__ == "__main__":

    env = SubprocVecEnv([make_env(i) for i in range(4)])  
    env = VecMonitor(env)

    agent1 = PPO.load("Agent1_4Game", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/red1/")
    agent2 = PPO.load("Agent2_4Game", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/red2/")
    agent3 = PPO.load("Agent3_4Game", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/blue1/")
    agent4 = PPO.load("Agent4_4Game", env=env, verbose=1, tensorboard_log="./ppo_tensorboard/blue2/")

    reset_once = True
    # Training loop
    for iteration in range(100000):

        env.env_method("set_current", 0)
        agent1.learn(total_timesteps=250_000, reset_num_timesteps=reset_once, callback=ActionHistogramCallback(agent_id=0))
        agent1.save("Agent1_4Game") 
        env.env_method("reload_agents")

        env.env_method("set_current", 1)
        agent2.learn(total_timesteps=250_000, reset_num_timesteps=reset_once, callback=ActionHistogramCallback(agent_id=1))
        agent2.save("Agent2_4Game")  
        env.env_method("reload_agents")

        env.env_method("set_current", 2)
        agent3.learn(total_timesteps=250_000, reset_num_timesteps=reset_once, callback=ActionHistogramCallback(agent_id=2))
        agent3.save("Agent3_4Game")
        env.env_method("reload_agents")

        env.env_method("set_current", 3)
        agent4.learn(total_timesteps=250_000, reset_num_timesteps=reset_once, callback=ActionHistogramCallback(agent_id=3))
        agent4.save("Agent4_4Game")
        env.env_method("reload_agents")

        print(f"Iteration {iteration + 1} complete.")
        reset_once = False
    
    
    print("Training complete.")

    #tensorboard --logdir file:///Users/xingsun/LunarLanding/logs
    #/Users/xingsun/Downloads/events.out.tfevents.1746838080.SL-2ZN2SL3.8560.0