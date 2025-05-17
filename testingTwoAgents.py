import gymnasium as gym
import pygame
import time

# Initialize Pygame
pygame.init()
pygame.display.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Lunar Lander - Two Agents")

# Create the environment
from GymLunarLander import GymLunarLander
env = GymLunarLander(render_mode="human")
raw_env = env.unwrapped  # Bypass Gymnasium's wrappers

try:
    obs, info = raw_env.reset()
    print("Controls:")
    print("Agent 0: LEFT=left thruster, UP=main engine, RIGHT=right thruster")
    print("Agent 1: A=left, W=up, D=right")
    print("Press ESC to quit")

    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard state for BOTH agents
        keys = pygame.key.get_pressed()
        
        # Agent 0 controls (arrow keys)
        action0 = 0  # No action
        if keys[pygame.K_LEFT]: action0 = 1
        elif keys[pygame.K_UP]: action0 = 2
        elif keys[pygame.K_RIGHT]: action0 = 3
        elif keys[pygame.K_DOWN]: action0 = 4
        
        # Step the environment with BOTH actions
        obs, reward, done, truncated, info = raw_env.step(0, 0)
        obs, reward, done, truncated, info = raw_env.step(action0, 1)
        obs, reward, done, truncated, info = raw_env.step(0, 2)
        obs, reward, done, truncated, info = raw_env.step(0, 3)
    
        raw_env.render()
        
        if done:
            print(f"Episode finished! Reward: {reward}")
            obs, info = raw_env.reset()
            pass
        
        clock.tick(60)  # 60 FPS
        
finally:
    raw_env.close()
    pygame.quit()