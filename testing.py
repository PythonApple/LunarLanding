import gymnasium as gym
from stable_baselines3 import PPO
from square_lunar_lander import SquareLunarLander
import pygame
import time

def run_trained_agent():
    # Initialize environment
    env = SquareLunarLander(render_mode='human')
    
    # Load trained model
    try:
        model = PPO.load("square_lander_basic")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run the agent
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Handle Pygame events (critical!)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get action from trained agent
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, _, done, _, _ = env.step(action)
        
        # Render
        env.render()
        
        # Control speed
        clock.tick(60)  # 60 FPS
        
        if done:
            print("Episode completed, resetting...")
            obs, _ = env.reset()
    
    env.close()

if __name__ == "__main__":
    run_trained_agent()