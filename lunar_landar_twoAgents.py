# square_lunar_lander.py
import gymnasium as gym
from gymnasium import spaces, Env
import numpy as np
import pygame
from Box2D import b2World, b2PolygonShape, b2Vec2
from Box2D.b2 import chainShape, contactListener
import math

class CollisionDetector(contactListener):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.ground_contacts = [False, False]
    
    def BeginContact(self, contact):
        bodyA, bodyB = contact.fixtureA.body, contact.fixtureB.body

        for i, rocket in enumerate(self.env.rockets):
            if (bodyA == rocket and bodyB == self.env.ground_body) or \
               (bodyB == rocket and bodyA == self.env.ground_body):
                #self.ground_contacts[i] = True
                pass
        

    def EndContact(self, contact):
        # Optional: Track when contact ends
        pass


class SquareLunarLanderTwo(Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    
    # Screen dimensions (pixels)
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    
    # Physics to pixels conversion
    PIXELS_PER_METER = 30
    
    # Rocket dimensions (physics units)
    ROCKET_WIDTH = 4   # meters (physics)
    ROCKET_HEIGHT = 4  # meters (physics)
    
    # Ground dimensions (physics units)
    GROUND_WIDTH = 50  # meters (physics)
    GROUND_HEIGHT = 1  # meters (physics)
    
    def __init__(self, render_mode=None, num_agents=2):

        self.num_agents = num_agents
        self.rockets = []

        # Collision detection
        self.ground_contact = False

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: No thrust, 1: Left, 2: Main, 3: Right
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        # Physics world
        self.world = b2World(gravity=(0, -10))
        self.world.contactListener = CollisionDetector(self)
        self._create_world()
        self._generate_terrain()

        # Rendering
        self.render_mode = render_mode
        pygame.init()
        if self.render_mode is not None:
            pygame.display.set_mode((1,1), pygame.HIDDEN)
            self.rocket_img1 = pygame.image.load("Soviet.png").convert_alpha()
            self.rocket_img2 = pygame.image.load("USA.png").convert_alpha()
            self.flame_img = pygame.image.load("flame.png").convert_alpha()
            self.flame_img = pygame.transform.scale(self.flame_img, (int(self.ROCKET_WIDTH * self.PIXELS_PER_METER), int(self.ROCKET_HEIGHT * self.PIXELS_PER_METER)))
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            else:
                self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()



    def _create_world(self):

        """Create physics bodies with labeled dimensions"""
        # Rocket body (physics coordinates)
        rocket1 = self.world.CreateDynamicBody(position=(-19, 27))
        rocket1.CreateFixture(
            shape=b2PolygonShape(box=(self.ROCKET_WIDTH/4, self.ROCKET_HEIGHT/4)),
            density=1,
            friction=0.3
        )
        rocket1.inertia = 100

        rocket2 = self.world.CreateDynamicBody(position=(19, 27))
        rocket2.CreateFixture(
            shape=b2PolygonShape(box=(self.ROCKET_WIDTH/4, self.ROCKET_HEIGHT/4)),
            density=1,
            friction=0.3
        )
        rocket2.inertia = 100
        self.rockets = [rocket1, rocket2]


        
        # Ground body (physics coordinates)
        self.ground = self.world.CreateStaticBody(position=(0, 0))
        self.ground.CreateFixture(
            shape=b2PolygonShape(box=(self.GROUND_WIDTH, self.GROUND_HEIGHT)),
            friction=0.5
        )

    def _generate_terrain(self):
        
        # Terrain parameters
        num_points = 20
        max_bump = 7  # meters
        segment_length = self.GROUND_WIDTH * 2 / num_points
        
        # Generate terrain vertices
        vertices = []
        for i in range(num_points + 1):
            x = -self.GROUND_WIDTH + i * segment_length
            y = max_bump * (np.random.random()) if 0 < i < num_points else 1
            if ( i  == 10):
                y = 1
            vertices.append((x, y))
        
        # Create chain shape (correct approach)
        self.ground_body = self.world.CreateStaticBody(position=(0, 0))
        self.ground_body.CreateFixture(
            shape=chainShape(vertices=vertices),  # Note lowercase 'c'
            friction=0.7,
            density=0
        )
        
        # Store for rendering
        self.ground_vertices = vertices

        # Flag
        self.flag_position = (0, 3)

    def physics_to_screen(self, physics_x, physics_y):
        """Convert Box2D coordinates to screen coordinates."""
        screen_x = physics_x * self.PIXELS_PER_METER + self.SCREEN_WIDTH // 2
        screen_y = self.SCREEN_HEIGHT - (physics_y * self.PIXELS_PER_METER)
        return (int(screen_x), int(screen_y))

    def _update_flame_state(self, rocket_index, direction):
        if direction is None or self.render_mode is None:
            setattr(self, f'active_flame_{rocket_index}', None)
            return
        
        rocket = self.rockets[rocket_index]
            
        # Normalize angle to 0-6 range (where 3 is upside down)
        normalized_angle = rocket.angle % 6.0
        
        # Calculate base offsets (before rotation)
        if direction == "right":
            offset_x = 1
            offset_y = 0
            # Flame should point right (+90° from rocket angle)
            flame_angle = normalized_angle - 1.5  # 1.5 = 90° in your 6=360° system
        elif direction == "left":
            offset_x = -1
            offset_y = 0
            # Flame should point left (-90° from rocket angle)
            flame_angle = normalized_angle + 1.5
        else:  # main engine
            offset_x = 0
            offset_y = -self.ROCKET_HEIGHT * 0.5
            # Flame always points opposite rocket bottom
            flame_angle = normalized_angle   # 180° flip
        
        # Convert angle to degrees for pygame (0-6 → 0-360°)
        pygame_angle = (flame_angle % 6.0) * 60  # 60° per unit
        
        # Rotate the flame image
        rotated_flame = pygame.transform.rotate(
            self.flame_img,
            -pygame_angle  # Negative because pygame's y-axis is inverted
        )
        
        # Calculate world position
        flame_world_x = rocket.position.x + offset_x
        flame_world_y = rocket.position.y + offset_y
        
        # Convert to screen coordinates and draw
        screen_pos = self.physics_to_screen(flame_world_x, flame_world_y)
        flame_rect = rotated_flame.get_rect(center=screen_pos)
        setattr(self, f'active_flame_{rocket_index}', {
            "rotated": rotated_flame,
            "rectangle": flame_rect
        })

    def step(self, action, current_agent):
        other_agent = 1 - current_agent

        self.world.contactListener.ground_contacts = [False, False]
        
        rocket = self.rockets[current_agent]

        velocity = rocket.linearVelocity.y

        if action == 1:  # Left thrust
            rocket.ApplyForce(b2Vec2(-200, 0), rocket.worldCenter-(0,1) , True)
            self._update_flame_state(current_agent, "right")

        elif action == 2:  # Main thrust (angle-dependent)
            rocket.ApplyForceToCenter((0,300), True)
            self._update_flame_state(current_agent, "main")

        elif action == 3:  # Right thrust
            rocket.ApplyForce(b2Vec2(200, 0), rocket.worldCenter-(0,1) , True)
            self._update_flame_state(current_agent, "left")
        else:
            self._update_flame_state(current_agent, None)
        
        self.world.Step(1/60, 6, 2)
        
        obs = np.array([
            self.rockets[current_agent].position.x,
            self.rockets[current_agent].position.y,
            self.rockets[current_agent].linearVelocity.x,
            self.rockets[current_agent].linearVelocity.y,
            self.rockets[current_agent].angle,
            self.rockets[other_agent].position.x,
            self.rockets[other_agent].position.y,
            self.rockets[other_agent].linearVelocity.x,
            self.rockets[other_agent].linearVelocity.y,
            self.rockets[other_agent].angle,
        ], dtype=np.float32)
       
        done = False
        reward = -0.1
        reward -= min(abs(self.rockets[current_agent].position.x - 0) * 0.1, 5)

        if self.world.contactListener.ground_contacts[current_agent]:
            reward += 150
            reward -= abs(velocity) * 4 
            reward -= abs(rocket.angle-6) * 5
            done = True
        elif abs(rocket.position.x) > 30:
            reward = -100
            done = True
          
        elif (rocket.position.y) > 35:
            reward = -100
            done = True
      
        #print(f"reward: {reward:.2f}")
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode is None:
            return None
            
        self.screen.blit(pygame.transform.smoothscale(pygame.image.load("background.png"), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)), (0, 0))
            
        # Draw bumpy ground
        if hasattr(self, 'ground_vertices'):
            screen_points = [
                self.physics_to_screen(x, y)
                for x, y in self.ground_vertices
            ]
            pygame.draw.lines(
                self.screen,
                (100, 100, 100),  # Ground color
                False,  # Not closed
                screen_points,
                3  # Line thickness
            )
            
            # Draw ground "thickness"
            bottom_y = self.physics_to_screen(0, -5)[1]  # Extend 5m down
            screen_points.append((screen_points[-1][0], bottom_y))
            screen_points.insert(0, (screen_points[0][0], bottom_y))
            pygame.draw.polygon(
                self.screen,
                (80, 80, 80),  # Darker fill color
                screen_points
            )

            # Draw Flag
            flag_pos = self.physics_to_screen(*self.flag_position)
            pygame.draw.line(
                self.screen, (255,255,255),
                flag_pos,
                (flag_pos[0], flag_pos[1] +60),  # 50px tall flagpole
                2
            )
            pygame.draw.polygon(
                self.screen, (255,0,0),
                [(flag_pos[0], flag_pos[1]-40),
                (flag_pos[0]+30, flag_pos[1]-20),
                (flag_pos[0], flag_pos[1])]
            )
            if hasattr(self, 'active_flame_0') and self.active_flame_0:
                self.screen.blit(self.active_flame_0["rotated"], self.active_flame_0["rectangle"])
            if hasattr(self, 'active_flame_1') and self.active_flame_1:
                self.screen.blit(self.active_flame_1["rotated"], self.active_flame_1["rectangle"])
            
            # Draw rocket
            self.rocket_img1 = pygame.transform.scale(self.rocket_img1, (int(self.ROCKET_WIDTH * self.PIXELS_PER_METER), int(self.ROCKET_HEIGHT * self.PIXELS_PER_METER)))
            self.rocket_img2 = pygame.transform.scale(self.rocket_img2, (int(self.ROCKET_WIDTH * self.PIXELS_PER_METER), int(self.ROCKET_HEIGHT * self.PIXELS_PER_METER)))
            # Rotate the rocket image based on its angle
            rotated_img1 = pygame.transform.rotate(self.rocket_img1, -np.degrees(self.rockets[0].angle))
            img_rect1 = rotated_img1.get_rect(center=self.physics_to_screen(self.rockets[0].position.x, self.rockets[0].position.y))
            self.screen.blit(rotated_img1, img_rect1.topleft)

            rotated_img2 = pygame.transform.rotate(self.rocket_img2, -np.degrees(self.rockets[1].angle))
            img_rect2 = rotated_img2.get_rect(center=self.physics_to_screen(self.rockets[1].position.x, self.rockets[1].position.y))
            self.screen.blit(rotated_img2, img_rect2.topleft)


            
            
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


    def reset(self, seed=None, options=None):
        # Destroy all existing rockets
        if hasattr(self, 'rockets'):
            for rocket in self.rockets:
                if rocket and rocket in self.world.bodies:  # Safety check
                    self.world.DestroyBody(rocket)
            self.rockets.clear()  # Empty the list
        
        # Destroy ground body if it exists
        if hasattr(self, 'ground_body') and self.ground_body in self.world.bodies:
            self.world.DestroyBody(self.ground_body)
        
        # Recreate the world
        self._create_world()  # This should repopulate self.rockets
        self._generate_terrain()
        
        return np.zeros(10, dtype=np.float32), {}


    def close(self):
        if self.screen is not None:
            pygame.quit()

# Registration
from gymnasium.envs.registration import register
register(
    id='SquareLunarLander-twoAgents-v0',
    entry_point='square_lunar_lander:SquareLunarLander',
)