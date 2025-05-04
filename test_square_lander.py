import gymnasium as gym
import pygame
import time

# 1. FIRST initialize Pygame properly
pygame.init()
pygame.display.init()  # This is the crucial line
screen = pygame.display.set_mode((800, 600))  # Temporary window
pygame.display.set_caption("Lunar Lander")

# 2. NOW create the environment
from square_lunar_lander import SquareLunarLander
env = gym.make('SquareLunarLander-v0', render_mode='human')

try:
    obs, info = env.reset()
    print("Controls: LEFT=left thruster, UP=main engine, RIGHT=right thruster")
    print("Press ESC or close window to quit")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events - this will now work
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        action = 0
        if keys[pygame.K_LEFT]: action = 1
        elif keys[pygame.K_UP]: action = 2
        elif keys[pygame.K_RIGHT]: action = 3
        
        # Environment step
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done:
            print(f"Reward: {reward:.2f}")
            obs, info = env.reset()
        
        # Control speed
        clock.tick(60)
        
finally:
    env.close()
    pygame.quit()