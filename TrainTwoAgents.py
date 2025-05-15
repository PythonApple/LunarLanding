from GymLunarLander import GymLunarLander
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.wrappers import RecordVideo
import os
import warnings


class RLlibWrapper(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.env = GymLunarLander(render_mode="human")
        #self.env = RecordVideo(self.env, "videos", episode_trigger=lambda x: True)
        
        self.agents = self.possible_agents = ["red1", "red2", "blue1", "blue2"]

        self.observation_spaces = {
            "red1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
            "red2": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
            "blue1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
            "blue2": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
        }
        self.action_spaces = {
            "red1": gym.spaces.Discrete(5),
            "red2": gym.spaces.Discrete(5),
            "blue1": gym.spaces.Discrete(5),
            "blue2": gym.spaces.Discrete(5),
        }

        self.count =0


    def reset(self, *, seed=None, options=None):
         self.count =0
         return self.env.reset()
    

    def step(self, action_dict):
        self.count +=1
        terminateds = {"__all__": False}
        rewards = {}
        team1_rewards =0
        team2_rewards =0

        obs, reward, terminated, truncated, info = self.env.step(action_dict["red1"], 0)
        team1_rewards += reward
        obs, reward, terminated, truncated, info = self.env.step(action_dict["red2"], 1)
        team1_rewards += reward
        obs, reward, terminated, truncated, info = self.env.step(0, 2)
        team2_rewards += reward
        obs, reward, terminated, truncated, info = self.env.step(0, 3)
        team2_rewards += reward

        rewards["red1"] = team1_rewards #- team2_rewards
        rewards["red2"] = team1_rewards #- team2_rewards
        rewards["blue1"] = team2_rewards -team1_rewards
        rewards["blue2"] = team2_rewards -team1_rewards

        if terminated or self.count >500:
            terminateds["__all__"] = True

        return(
            {"red1": np.array(obs + [1,0,0,0], dtype=np.float32), 
             "red2": np.array(obs + [0,1,0,0], dtype=np.float32), 
             "blue1": np.array(obs + [0,0,1,0], dtype=np.float32), 
             "blue2": np.array(obs + [0,0,0,1], dtype=np.float32), 
            },
            rewards,
            terminateds,
            {},
            {},
        )
    

config = (
    PPOConfig()
    .environment(env = RLlibWrapper)
    .env_runners(num_env_runners=1, rollout_fragment_length=512)
    .framework("torch", 
               torch_compile_learner=True, 
               torch_compile_learner_dynamo_backend="inductor",
               torch_compile_learner_dynamo_mode="default")
    .training(
        train_batch_size=2048,
        num_sgd_iter=10,
        gamma=0.99,
        lr=3e-4,
        grad_clip=0.5,
    )
    .multi_agent(
        policies={
            "redTeam": (None, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32), gym.spaces.Discrete(5), {}),
            "blueTeam": (None, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32), gym.spaces.Discrete(5), {}),
        },
        policy_mapping_fn=lambda agent_id, info: ("redTeam" if agent_id.startswith("red") else "blueTeam"),
        policies_to_train=["redTeam"], 
    )
)
config.model["fcnet_hiddens"] = [64, 64]
algo = config.build()
algo.restore("/Users/xingsun/LunarLanding/checkpoints/red_blue_model")
save_dir = os.path.abspath("checkpoints/red_blue_model")


for iteration in range(100000):  #
    result = algo.train()
    if iteration % 20 == 0:
        print(f"Iteration {iteration}")
        print(f"result: {result}")
        algo.save(f"file://{save_dir}")


