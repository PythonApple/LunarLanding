from stable_baselines3 import PPO
from GymLunarLander import GymLunarLander
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


class CompetitiveEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.agents = []
        self.agents.append(PPO.load("Agent1_4Game"))
        self.agents.append(PPO.load("Agent2_4Game"))
        self.agents.append(PPO.load("Agent3_4Game"))
        self.agents.append(PPO.load("Agent4_4Game"))
        self.current_agent = None

    def set_current(self, agent_id):
        self.current_agent = agent_id
 
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
      

        if self.current_agent in [0,1]:
            return obs, team1_score - team2_score, done, truncated, info
        elif self.current_agent in [2,3]:
            return obs, team2_score - team1_score, done, truncated, info
            
def make_env():
    def _init():
        base_env = GymLunarLander(render_mode="rgb_array")
        CompEnv = CompetitiveEnvWrapper(base_env)
        timelimit_env = TimeLimit(CompEnv, max_episode_steps=1000)  # 1000 steps per episode
        env = RecordVideo(
            timelimit_env,
            video_folder="./4Game3",
            episode_trigger=lambda x: x % 10 == 0,  # Record every 100 episodes
            disable_logger=True
        )
        return env
    return _init

if __name__ == "__main__":

    env = SubprocVecEnv([make_env() for _ in range(4)])  
    env = VecMonitor(env)

    agent1 = PPO.load("Agent1_4Game", env=env, verbose=1)
    agent2 = PPO.load("Agent2_4Game", env=env, verbose=1)
    agent3_loaded = PPO.load("Agent3_4Game", env=env, verbose=1)
    agent4_loaded = PPO.load("Agent4_4Game", env=env, verbose=1)

    agent3 = PPO(
    "MlpPolicy",
    env=env,
    learning_rate=1e-3,
    ent_coef=0.02,
    vf_coef=0.25,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    clip_range=lambda _: 0.2,
    verbose=1,
    )
    agent3.policy.load_state_dict(agent3_loaded.policy.state_dict())

    agent4 = PPO(
    "MlpPolicy",
    env=env,
    learning_rate=1e-3,
    ent_coef=0.02,
    vf_coef=0.25,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    clip_range=lambda _: 0.2,
    verbose=1,
    )
    agent4.policy.load_state_dict(agent4_loaded.policy.state_dict())

    # Training loop
    for iteration in range(100000):

        #env.env_method("set_current", 0)
        #agent1.learn(total_timesteps=10_000, reset_num_timesteps=False)
        #env.env_method("set_current", 1)
        #agent2.learn(total_timesteps=10_000, reset_num_timesteps=False)
        env.env_method("set_current", 2)
        agent3.learn(total_timesteps=100_000, reset_num_timesteps=True)
        env.env_method("set_current", 3)
        agent4.learn(total_timesteps=100_000, reset_num_timesteps=True)

        agent1.save("Agent1_4Game")  
        agent2.save("Agent2_4Game")  
        agent3.save("Agent3_4Game")
        agent4.save("Agent4_4Game")

        print(f"Iteration {iteration + 1} complete.")
    print("Training complete.")

    #tensorboard --logdir file:///Users/xingsun/LunarLanding/logs
    #/Users/xingsun/Downloads/events.out.tfevents.1746838080.SL-2ZN2SL3.8560.0