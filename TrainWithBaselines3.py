from stable_baselines3 import PPO
from GymLunarLander import GymLunarLander
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import numpy as np
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import os
from stable_baselines3.common.logger import TensorBoardOutputFormat


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

        team1_score /= 5
        team2_score /= 5
      
        self.env.set_score(team2_score-team1_score)

        if self.current_agent in [0,1]:
            return obs, team1_score - team2_score, done, truncated, info
        elif self.current_agent in [2,3]:
            return obs, team2_score - team1_score, done, truncated, info

class SpeedLoggerCallback(BaseCallback):
    def __init__(self, agent_id, obs_stride = 8, verbose=0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.obs_stride = obs_stride
        self.total_speed = 0.0
        self.steps = 0

    def _on_rollout_start(self):
        self.total_speed = 0.0
        self.steps = 0

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"][0]  # unwrap VecEnv
        offset = self.agent_id * self.obs_stride
        vx = obs[offset + 2]
        vy = obs[offset + 3]

        speed = np.sqrt(vx**2 + vy**2)
        self.total_speed += speed
        self.steps += 1
        return True

    def _on_rollout_end(self):
        if self.steps > 0:
            avg_speed = self.total_speed / self.steps
            self.logger.record(f"agent{self.agent_id}/avg_speed", avg_speed)

class PositionHeatmapCallback(BaseCallback):
    def __init__(self, agent_id, verbose=0, log_dir="./ppo_tensorboard/", bins=50, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.log_dir = log_dir
        self.bins = bins
        self.xlim = xlim
        self.ylim = ylim
        self.writer = None
        self.xs = []
        self.ys = []
        self.num_rollouts = 0

    def _on_training_start(self) -> None:
        run_path = os.path.join(self.log_dir, f"agent{self.agent_id}")
        os.makedirs(run_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=run_path)

    def _on_step(self) -> bool:
        offset = self.agent_id * 8
        obs = self.locals["new_obs"]
        
        # Handle vectorized env
        if len(obs.shape) == 2:
            for i in range(obs.shape[0]):
                x = obs[i][0 + offset]
                y = obs[i][1 + offset]
                self.xs.append(x)
                self.ys.append(y)
        else:
            x = obs[0 + offset]
            y = obs[1 + offset]
            self.xs.append(x)
            self.ys.append(y)

        return True

    def _on_rollout_end(self) -> None:
        self.num_rollouts += 1
        if len(self.xs) == 0:
            return

        heatmap, xedges, yedges = np.histogram2d(
            self.xs, self.ys, bins=self.bins, range=[self.xlim, self.ylim]
        )

        heatmap = heatmap.T
        heatmap /= np.max(heatmap) + 1e-8

        img = (heatmap * 255).astype(np.uint8)
        img = np.expand_dims(img, axis=2)  # (H, W, 1)
        img = np.repeat(img, 3, axis=2)    # (H, W, 3)
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

        if self.writer:
            self.writer.add_image(
                tag=f"agent{self.agent_id}/position_heatmap",
                img_tensor=img,
                global_step=self.num_timesteps,
                dataformats="NHWC"
            )

        self.xs.clear()
        self.ys.clear()

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()

class ActionHistogramCallback(BaseCallback):
    def __init__(self, agent_id, verbose=0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.actions = []

    def _on_step(self) -> bool:
        action = self.locals.get("actions", None)
        if action is not None:
            if isinstance(action, np.ndarray):
                self.actions.extend(action.tolist())
            elif isinstance(action, list):
                self.actions.extend(action)
            else:
                self.actions.append(action)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.actions) == 0:
            return

        actions_array = np.array(self.actions)

        # --- Find the TensorBoard writer ---
        writer = None
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                writer = fmt.writer
                break

        # --- Log histogram if writer is available ---
        if writer is not None:
            writer.add_histogram(
                tag=f"agent{self.agent_id}/actions_histogram",
                values=actions_array,
                global_step=self.num_timesteps
            )

        self.actions.clear()

def make_env(id):
    def _init():
        base_env = GymLunarLander(render_mode="rgb_array")
        CompEnv = CompetitiveEnvWrapper(base_env)
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

    # Training loop
    for iteration in range(100000):

        env.env_method("set_current", 0)
        agent1.learn(total_timesteps=250_000, reset_num_timesteps=False, callback=CallbackList([SpeedLoggerCallback(agent_id=0), PositionHeatmapCallback(agent_id=0), ActionHistogramCallback(agent_id=0)]))
        env.env_method("set_current", 1)
        agent2.learn(total_timesteps=250_000, reset_num_timesteps=False, callback=CallbackList([SpeedLoggerCallback(agent_id=1), PositionHeatmapCallback(agent_id=1), ActionHistogramCallback(agent_id=1)]))
        env.env_method("set_current", 2)
        agent3.learn(total_timesteps=250_000, reset_num_timesteps=False, callback=CallbackList([SpeedLoggerCallback(agent_id=2), PositionHeatmapCallback(agent_id=2), ActionHistogramCallback(agent_id=2)]))
        env.env_method("set_current", 3)
        agent4.learn(total_timesteps=250_000, reset_num_timesteps=False, callback=CallbackList([SpeedLoggerCallback(agent_id=3), PositionHeatmapCallback(agent_id=3), ActionHistogramCallback(agent_id=3)]))

        agent1.save("Agent1_4Game")  
        agent2.save("Agent2_4Game")  
        agent3.save("Agent3_4Game")
        agent4.save("Agent4_4Game")

        print(f"Iteration {iteration + 1} complete.")
    print("Training complete.")

    #tensorboard --logdir file:///Users/xingsun/LunarLanding/logs
    #/Users/xingsun/Downloads/events.out.tfevents.1746838080.SL-2ZN2SL3.8560.0