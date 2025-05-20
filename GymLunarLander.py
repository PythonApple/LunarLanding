__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
from gymnasium.utils.step_api_compatibility import step_api_compatibility


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame


FPS = 100
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 52.0
SIDE_ENGINE_POWER = 2.4

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 1200
VIEWPORT_H = 800


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body
        
        # Check for lander-moon collisions (game over)
        for index, lander in enumerate(self.env.landers):
            if (lander == bodyA and bodyB == self.env.moon) or \
               (lander == bodyB and bodyA == self.env.moon):
                self.env.agent_crashed[index] = True 
                self.env.dones[index] = True
             
        
        # Check for leg-moon contacts
        for lander_legs in self.env.legs:
            for leg in lander_legs:
                if (leg == bodyA and bodyB == self.env.moon) or \
                   (leg == bodyB and bodyA == self.env.moon):
                    leg.ground_contact = True
           

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body
        
        # Check for lander-moon collisions (game over)
        for index, lander in enumerate(self.env.landers):
            if (lander == bodyA and bodyB == self.env.moon) or \
               (lander == bodyB and bodyA == self.env.moon):
                self.env.agent_crashed[index] = False
                self.env.dones[index] = False
             
        
        # Check for leg-moon contacts
        for lander_legs in self.env.legs:
            for leg in lander_legs:
                if (leg == bodyA and bodyB == self.env.moon) or \
                   (leg == bodyB and bodyA == self.env.moon):
                    leg.ground_contact = False

class GymLunarLander(gym.Env, EzPickle):
    r"""
    ## Description
    This environment is a classic rocket trajectory optimization problem.
    According to Pontryagin's maximum principle, it is optimal to fire the
    engine at full throttle or turn it off. This is the reason why this
    environment has discrete actions: engine on or off.

    There are two environment versions: discrete or continuous.
    The landing pad is always at coordinates (0,0). The coordinates are the
    first two numbers in the state vector.
    Landing outside of the landing pad is possible. Fuel is infinite, so an agent
    can learn to fly and then land on its first attempt.

    To see a heuristic landing, run:
    ```shell
    python gymnasium/envs/box2d/lunar_lander.py
    ```

    ## Action Space
    There are four discrete actions available:
    - 0: do nothing
    - 1: fire left orientation engine
    - 2: fire main engine
    - 3: fire right orientation engine

    ## Observation Space
    The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
    velocities in `x` & `y`, its angle, its angular velocity, and two booleans
    that represent whether each leg is in contact with the ground or not.

    ## Rewards
    After every step a reward is granted. The total reward of an episode is the
    sum of the rewards for all the steps within that episode.

    For each step, the reward:
    - is increased/decreased the closer/further the lander is to the landing pad.
    - is increased/decreased the slower/faster the lander is moving.
    - is decreased the more the lander is tilted (angle not horizontal).
    - is increased by 10 points for each leg that is in contact with the ground.
    - is decreased by 0.03 points each frame a side engine is firing.
    - is decreased by 0.3 points each frame the main engine is firing.

    The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

    An episode is considered a solution if it scores at least 200 points.

    ## Starting State
    The lander starts at the top center of the viewport with a random initial
    force applied to its center of mass.

    ## Episode Termination
    The episode finishes if:
    1) the lander crashes (the lander body gets in contact with the moon);
    2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
    3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md__d_1__git_hub_box2d_docs_dynamics.html#autotoc_md61),
        a body which is not awake is a body which doesn't move and doesn't
        collide with any other body:
    > When Box2D determines that a body (or group of bodies) has come to rest,
    > the body enters a sleep state which has very little CPU overhead. If a
    > body is awake and collides with a sleeping body, then the sleeping body
    > wakes up. Bodies will also wake up if a joint or contact attached to
    > them is destroyed.

    ## Arguments

    Lunar Lander has a large number of arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
    ...                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>

    ```

     * `continuous` determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used with the
     action space being `Discrete(4)` or `Box(-1, +1, (2,), dtype=np.float32)` respectively.
     For continuous actions, the first coordinate of an action determines the throttle of the main engine, while the second
     coordinate specifies the throttle of the lateral boosters. Given an action `np.array([main, lateral])`, the main
     engine will be turned off completely if `main < 0` and the throttle scales affinely from 50% to 100% for
     `0 <= main <= 1` (in particular, the main engine doesn't work  with less than 50% power).
     Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
     booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
     from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

    * `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12. Default is -10.0

    * `enable_wind` determines if there will be wind effects applied to the lander. The wind is generated using
     the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))` where `k` is set to 0.01 and `C` is sampled randomly between -9999 and 9999.

    * `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for
     `wind_power` is between 0.0 and 20.0.

    * `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.
     The recommended value for `turbulence_power` is between 0.0 and 2.0.

    ## Version History
    - v3:
        - Reset wind and turbulence offset (`C`) whenever the environment is reset to ensure statistical independence between consecutive episodes (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/954)).
        - Fix non-deterministic behaviour due to not fully destroying the world (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/728)).
        - Changed observation space for `x`, `y`  coordinates from $\pm 1.5$ to $\pm 2.5$, velocities from $\pm 5$ to $\pm 10$ and angles from $\pm \pi$ to $\pm 2\pi$ (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/752)).
    - v2: Count energy spent and in v0.24, added turbulence with wind power and turbulence_power parameters
    - v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.
    - v0: Initial version

    ## Notes

    There are several unexpected bugs with the implementation of the environment.

    1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
    This in turn results in an orientation dependent torque being applied to the lander.

    2. The units of the state are not consistent. I.e.
    * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.

    For the default values of VIEWPORT_W, VIEWPORT_H, SCALE, and FPS, the scale factors equal:
    'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5

    After the correction has been made, the units of the state are as follows:
    'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)

    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        gravity: float = -10.0,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            gravity,
            wind_power,
            turbulence_power,
        )

        assert (
            -12.0 < gravity and gravity < 0.0
        ), f"gravity (current value: {gravity}) must be between -12 and 0"
        self.gravity = gravity

        if 0.0 > wind_power or wind_power > 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        self.wind_power = wind_power

        if 0.0 > turbulence_power or turbulence_power > 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )
        self.turbulence_power = turbulence_power

        self.screen: pygame.Surface = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.landers = []
        self.legs = []
        self.particles = []
        self.rockets = []
        self.dones = [False ,False, False, False]
        self.agent_crashed = [False, False, False, False]
        self.shapings = [None, None, None, None]
        self.score = 0

        self.prev_reward = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        self.action_space = spaces.Discrete(5)

        self.render_mode = render_mode


    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.landers[0])
        self.world.DestroyBody(self.landers[1])
        self.world.DestroyBody(self.landers[2])
        self.world.DestroyBody(self.landers[3])
        self.landers = []
        self.world.DestroyBody(self.legs[0][0])
        self.world.DestroyBody(self.legs[0][1])
        self.world.DestroyBody(self.legs[1][0])
        self.world.DestroyBody(self.legs[1][1])
        self.world.DestroyBody(self.legs[2][0])
        self.world.DestroyBody(self.legs[2][1])
        self.world.DestroyBody(self.legs[3][0])
        self.world.DestroyBody(self.legs[3][1])
        self.legs = []
        self.rockets = []
        self.dones = [False, False, False, False]
        self.agent_crashed = [False, False, False, False]
        self.shapings = [None, None, None, None]
        self.score =0
    
    def _create_lander(self, lander_x, lander_y, color1, color2):
        lander = self.world.CreateDynamicBody(
            position=(lander_x, lander_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                restitution=0.0,
            ), 
        )

        lander.lander_x = lander_x
        lander.lander_y = lander_y
        lander.color1 = color1
        lander.color2 = color2
        lander.allowSleep = False
        self.landers.append(lander)


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()

        # Bug's workaround for: https://github.com/Farama-Foundation/Gymnasium/issues/728
        # Not sure why the self._destroy() is not enough to clean(reset) the total world environment elements, need more investigation on the root cause,
        # we must create a totally new world for self.reset(), or the bug#728 will happen
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Create Terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        height[0] = H * 0.45
        height[1] = H * 0.35
        height[-1] = H * 0.45
        height[-2] = H * 0.35
        #height = np.full((CHUNKS + 1,), H / 4)  # or just use self.helipad_y if it's already defined

        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)


        lander1_y = VIEWPORT_H / SCALE * 0.9        #RED TEAM
        lander1_x = VIEWPORT_W / SCALE / 2 *.10  
        lander2_y = VIEWPORT_H / SCALE * 0.9
        lander2_x = VIEWPORT_W / SCALE / 2 *.25

        lander3_y = VIEWPORT_H / SCALE * 0.9        #BLUE TEAM
        lander3_x = VIEWPORT_W / SCALE *.87
        lander4_y = VIEWPORT_H / SCALE * 0.9
        lander4_x = VIEWPORT_W / SCALE *.81


        self._create_lander(lander1_x, lander1_y, (224, 49, 0), (224, 49, 0))
        self._create_lander(lander2_x, lander2_y, (220, 60, 60), (137, 30, 0)) 
        self._create_lander(lander3_x, lander3_y, (0, 209, 224), (0, 173, 230))
        self._create_lander(lander4_x, lander4_y, (70, 130, 180), (0, 107, 142)) 

        for lander in self.landers:
            lander_legs = []
            for i in [-1, +1]:
                leg = self.world.CreateDynamicBody(
                    position=(lander.lander_x - i * LEG_AWAY / SCALE, lander.lander_y),
                    angle=(i * 0.05),
                    fixtures=fixtureDef(
                        shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                        density=1.0,
                        restitution=0.0,
                        categoryBits=0x0020,
                        #maskBits=0x001,
                    ),
                )
                leg.ground_contact = False
                leg.color1 = lander.color1
                leg.color2 = lander.color2
                rjd = revoluteJointDef(
                    bodyA=lander,
                    bodyB=leg,
                    localAnchorA=(0, 0),
                    localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=LEG_SPRING_TORQUE,
                    motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
                )
                if i == -1:
                    rjd.lowerAngle = (
                        +0.9 - 0.5
                    )  # The most esoteric numbers here, angled legs have freedom to travel within
                    rjd.upperAngle = +0.9
                else:
                    rjd.lowerAngle = -0.9
                    rjd.upperAngle = -0.9 + 0.5
                leg.joint = self.world.CreateJoint(rjd)
                lander_legs.append(leg)
            self.legs.append(lander_legs)

        self.drawlist = []
        for lander in self.landers:
            self.drawlist.append(lander)  # Add lander
            for leg in self.legs[self.landers.index(lander)]:  # Add its legs
                self.drawlist.append(leg)

        if self.render_mode == "human":
            self.render()

        #return {
        #    "red1": np.random.rand(36).astype(np.float32),
        #    "red2": np.random.rand(36).astype(np.float32),
        #    "blue1": np.random.rand(36).astype(np.float32),
        #    "blue2": np.random.rand(36).astype(np.float32),
        #}, {}
        return self.step(0, 0)[0], {} 

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))
        while self.rockets and (all_particle or self.rockets[0].ttl < 0):
            self.world.DestroyBody(self.rockets.pop(0))

    def _create_rocket(self, x, y, angle, speed=1): 

        triangle_vertices = [
            (-6/SCALE,0),
            (2/SCALE, -4/SCALE),  # Bottom right
            (2/SCALE, 4/SCALE)   # Top right
        ]
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        rocket = self.world.CreateDynamicBody(
            position = (x, y),
            angle=angle,
            linearVelocity = (vx, vy),
            fixtures = fixtureDef(
                shape = polygonShape(vertices = triangle_vertices),
                density = 1.0,
            ),
        )
        rocket.ttl= 15
        rocket.is_rocket = True
        rocket.is_triangle = True
        self.rockets.append(rocket)

    def set_score(self, score):
        self.score += score

    def step(self, action, curr_agent):
        lander = self.landers[curr_agent]

        # Update wind and apply to the lander
        assert lander is not None, "You forgot to call reset()"

        # Apply Engine Impulses

        # Tip is the (X and Y) components of the rotation of the lander.
        tip = (math.sin(lander.angle), math.cos(lander.angle))

        # Side is the (-Y and X) components of the rotation of the lander.
        side = (-tip[1], tip[0])

        # Generate two random numbers between -1/SCALE and 1/SCALE.
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        if (curr_agent == 1 or curr_agent == 3):
            if (action == 4):
                if (curr_agent == 3):
                    self._create_rocket(lander.position.x, lander.position.y, lander.angle + 3, speed=50.0)
                elif (curr_agent == 1):
                    self._create_rocket(lander.position.x, lander.position.y, lander.angle, speed=50.0)


        m_power = 0.0
        if (action == 2):

            m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            # The components of the impulse to be applied by the main engine.
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )

            impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(
                    3.5,  # 3.5 is here to make particle speed adequate
                    impulse_pos[0],
                    impulse_pos[1],
                    m_power,
                )
                p.ApplyLinearImpulse(
                    (
                        ox * MAIN_ENGINE_POWER * m_power,
                        oy * MAIN_ENGINE_POWER * m_power,
                    ),
                    impulse_pos,
                    True,
                )
            lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if ( action in [1, 3]):
            # Orientation/Side engines

            # action = 1 is left, action = 3 is right
            direction = action - 2
            s_power = 1.0

            # The components of the impulse to be applied by the side engines.
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )

            # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
            # However, SIDE_ENGINE_HEIGHT is defined as 14
            # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
            # This in turn results in an orientation dependent torque being applied to the lander.
            impulse_pos = (
                lander.position[0] + ox - tip[0] * 17 / SCALE,
                lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (
                        ox * SIDE_ENGINE_POWER * s_power,
                        oy * SIDE_ENGINE_POWER * s_power,
                    ),
                    impulse_pos,
                    True,
                )
            lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        POSX_CLAMP = (-1.5, 1.5)
        POSY_CLAMP = (-0.5, 1.7)
        VEL_CLAMP = (-5, 5)
        state = []
        for i in range(4):
            lander_temp = self.landers[i]
            leg1, leg2 = self.legs[i]
            state += [
                np.clip((lander_temp.position.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2), *POSX_CLAMP),
                np.clip((lander_temp.position.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2), *POSY_CLAMP),
                np.clip(lander_temp.linearVelocity.x * (VIEWPORT_W / SCALE / 2) / FPS, *VEL_CLAMP),
                np.clip(lander_temp.linearVelocity.y * (VIEWPORT_H / SCALE / 2) / FPS, *VEL_CLAMP),
                ((lander_temp.angle + math.pi) % (2 * math.pi)) - math.pi,
                np.clip(20.0 * lander_temp.angularVelocity / FPS, *VEL_CLAMP),
                1.0 if leg1.ground_contact else 0.0,
                1.0 if leg2.ground_contact else 0.0,
            ]
        assert len(state) == 32
        
        if (curr_agent == 0):
            offset = 0
        elif (curr_agent == 1):
            offset = 8
        elif (curr_agent == 2):
            offset = 16
        elif (curr_agent == 3):
            offset = 24

        reward = 0
        if curr_agent == 0 or curr_agent == 2:
            
            reward -= 10 * np.sqrt(state[0 + offset] * state[0 + offset] + state[1 + offset] * state[1 + offset])
            reward -= 15 * np.sqrt(state[2 + offset] * state[2 + offset] + state[3 + offset] * state[3 + offset])

            if state[6+offset] == True:
                reward += .5
            if state[7+offset] == True:
                reward += .5

        if abs(state[0 + offset]) >= 1.0 or abs(state[1 + offset]) >= 1.6:
            reward -= 20
        
        terminated = False

        if self.render_mode == "human":
            self.render()

        return state, reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )

        for obj in self.rockets:
            obj.ttl -= 0.15

        self._clean_particles(False)

        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )

                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color2, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color2)
                    pygame.draw.aalines(
                        self.surf, color=obj.color2, points=path, closed=True
                    )

                for x in [self.helipad_x1, self.helipad_x2]:
                    x = x * SCALE
                    flagy1 = self.helipad_y * SCALE
                    flagy2 = flagy1 + 50
                    pygame.draw.line(
                        self.surf,
                        color=(255, 255, 255),
                        start_pos=(x, flagy1),
                        end_pos=(x, flagy2),
                        width=1,
                    )
                    pygame.draw.polygon(
                        self.surf,
                        color=(204, 204, 0),
                        points=[
                            (x, flagy2),
                            (x, flagy2 - 10),
                            (x + 25, flagy2 - 5),
                        ],
                    )
                    gfxdraw.aapolygon(
                        self.surf,
                        [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                        (204, 204, 0),
                    )

        for rocket in self.rockets:
            if hasattr(rocket, 'is_triangle'):
                vertices = [rocket.GetWorldPoint(vertex)
                            for vertex in rocket.fixtures[0].shape.vertices
                            ]
                screen_vertices = [(v[0] * SCALE, v[1] * SCALE) for v in vertices]
                pygame.draw.polygon(self.surf, (255, 100, 0), screen_vertices)

        self.surf = pygame.transform.flip(self.surf, False, True)

        pygame.font.init()
        font = pygame.font.SysFont(None, 75)
        if self.score < 0:
            text_surf = font.render(str(abs(self.score)), True, (224, 49, 0))
        else:
            text_surf = font.render(str(self.score), True, (0, 173, 230))
        self.surf.blit(text_surf, (10,10))

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False




