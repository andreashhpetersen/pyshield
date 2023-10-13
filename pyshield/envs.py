import math
import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils

# gravity
G = -9.81


class BouncingBallEnv(gym.Env):
    metadata = { 'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(
        self, render_mode=None, ts_size=0.3,
        max_n_steps=400, obs_low=[0, -25], obs_high=[50, 25]
    ):

        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
        )
        self.action_space = gym.spaces.Discrete(2)

        self.ts = ts_size
        self.max_n_steps = max_n_steps
        self.render_mode = render_mode

    def _get_obs(self):
        return np.array([self.p, self.v], dtype=np.float32)

    def _get_info(self):
        return { 'time': self.time }

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)

        self.time = 0.0
        self.steps_taken = 0

        self.p = 7 + self.np_random.uniform(0, 3)
        self.v = 0.0

        # for rendering
        self.positions = []
        self.velocities = []
        self.actions = []

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def allowed_actions(self, s):
        p, v = s
        if not self.is_safe(s):
            return []

        if p < 4:
            return [0]

        return [0, 1]

    def is_safe(self, s):
        p, v = s
        return not (p <= 0.01 and abs(v) <= 1)

    def step_from(self, s, a):
        return self._step(s, a)

    def step(self, action):
        new_state, reward, terminated = self._step(self._get_obs(), action)

        self.time += self.ts
        self.steps_taken += 1
        truncated = self.steps_taken >= self.max_n_steps

        self.p, self.v = new_state

        self.positions.append(self.p)
        self.velocities.append(self.v)
        self.actions.append(action)

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _step(self, state, action):
        p, v = state

        terminated = False

        # if an action is taken and position is at least 4
        if action == 1 and p >= 4.0:

            # hit when ball is going up
            if v >= 0.0:
                flip = -(0.9 + self.np_random.uniform(0, 0.1))
                v = (flip * v) - 4.0

            # hit when ball is already falling
            elif v >= -4.0:
                v = -4.0

        # new candidate state
        new_v = v + (self.ts * G)
        new_p = p + (v * self.ts) + 0.5 * (G * np.square(self.ts))

        # ball bounces!
        if new_p <= 0.0 and v < 0.0:

            # solve for t when p == 0
            D = np.sqrt(np.square(v) - (2 * G * p))
            t = max((-v + D) / G, (-v - D) / G)

            # velocity when the ball hits the ground
            new_v = v + (t * G)

            # flip velocity at bounce and loose some momentum
            new_v *= -(0.85 + self.np_random.uniform(0, 0.12))
            if new_v <= 1:
                terminated = True
                new_v, new_p = 0, 0

            else:
                # new position (starting from 0 as we have just bounced)
                new_p = max(
                    0, new_v * (self.ts - t) + 0.5 * (G * np.square(self.ts - t))
                )

                # new velocity
                new_v = new_v + (self.ts - t) * G if new_p > 0 else 0

        reward = -1 * action - 1000 * terminated

        return (new_p, new_v), reward, terminated


class RandomWalkEnv(gym.Env):
    metadata = { 'render_modes': [] }
    SLOW = 0
    FAST = 1

    def __init__(self, render_mode=None, obs_low=[0., 1.], obs_high=[0., 1.], unlucky=False):
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(2)

        self.x_max = 1.
        self.t_max = 1.

        # the changes in x and t depending on action (fast or slow)
        self.dx_fast = 0.17
        self.dx_slow = 0.1
        self.dt_fast = 0.05
        self.dt_slow = 0.12

        # randomness
        self.eps = 0.04
        self.unlucky = unlucky

    def _get_obs(self):
        return [self.x, self.t]

    def _get_info(self):
        return {}

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)

        self.x = 0.
        self.t = 0.

        return self._get_obs(), self._get_info()

    def is_safe(self, s):
        x, t = s
        return t < self.t_max

    def allowed_actions(self, s):
        x, t = s
        if x < self.x_max and t < self.t_max:
            return [0, 1]
        else:
            return []

    def step_from(self, s, a):
        return self._step(s, a)

    def step(self, action):
        (x, t), reward, terminated = self._step(self._get_obs(), action)

        self.x = x
        self.t = t

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _step(self, state, action):
        x, t = state

        # calculate cost
        cost = 3 if action == self.FAST else 1

        # move (fast or slow)
        nx = x + (self.dx_fast if action == self.FAST else self.dx_slow)
        nt = t + (self.dt_fast if action == self.FAST else self.dt_slow)

        # apply randomness
        nx -= self.eps if self.unlucky else self.get_random()
        nt -= -self.eps if self.unlucky else self.get_random()

        # check if game is over
        terminated = False
        if nt >= self.t_max:
            cost += 20
            terminated = True

        elif nx >= self.x_max and nt < self.t_max:
            terminated = True

        return (nx, nt), -cost, terminated

    def get_random(self):
        return self.eps + np.random.random() * (self.eps * 2)


class CartPoleEnv(gym.Env):
    metadata = { 'render_modes': [] }

    def __init__(self, render_mode=None, obs_low=None, obs_high=None, unlucky=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        if obs_high is None:
            # Angle limit set to 2 * theta_threshold_radians so failing observation
            # is still within bounds.
            high = np.array(
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max,
                ],
                dtype=np.float32,
            )
        else:
            high = np.array(obs_high)

        if obs_low is None:
            low = - high
        else:
            low = np.array(obs_low)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.action_space = spaces.Discrete(2)

        self.render_mode = render_mode
        self.unlucky = unlucky

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def _get_obs(self):
        pass

    def _get_info(self):
        return {}

    def is_safe(self, state):
        x, _, theta, _ = state
        return not bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        assert self.state is not None, "Call reset before using step method."

        nstate, reward, terminated = self._step(self.state, action)
        self.state = nstate

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        return nstate, reward, terminated, False, self._get_info()

    def step_from(self, state, action):
        return self._step(state, action)

    def _step(self, state, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        nstate = (x, x_dot, theta, theta_dot)

        terminated = False if self.is_safe(nstate) else True

        return np.array(nstate, dtype=np.float32), 1.0, terminated

    def allowed_actions(self, state):
        return list(range(2)) if self.is_safe(state) else []

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
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class CruiseControlEnv(gym.Env):
    DEC = -2
    NEU = 0
    ACC = 2

    def __init__(
        self, render_mode=None, unlucky=False,
            obs_low=[-10., -8., 0.], obs_high=[20., 20., 200.]
    ):
        self.unlucky = unlucky

        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(3)
        self.action_to_acceleration = [self.DEC, self.NEU, self.ACC]

        # dynamics
        self.v_ego_max = 20.
        self.v_ego_min = -10.

        self.v_front_max = 20.
        self.v_front_min = -8.

        self.max_sensor_distance = 200

        # state
        self.v_ego = 0.
        self.v_front = 0.
        self.distance = 10.

        # track total distance
        self.total_distance = 10.

    def _get_obs(self):
        return [ self.v_ego, self.v_front, self.distance ]

    def _get_info(self):
        return {}

    def reset(self):
        self.v_ego = 0.
        self.v_front = 0.
        self.distance = 10.
        self.total_distance = 10.

    def is_safe(self, state):
        _, _, distance = state
        return distance > 0

    def allowed_actions(self, state):
        v_ego, _, _ = state

        if not self.is_safe(state):
            return []

        allowed_actions = [0]
        if v_ego > self.v_ego_min:
            allowed_actions.append(1)

        if v_ego < self.v_ego_max:
            allowed_actions.append(2)

        return allowed_actions

    def step_from(self, state, action):
        return self._step(state, action)

    def step(self, action):
        nstate, reward, terminated = self._step(self._get_obs(), action)
        v_ego, v_front, distance = nstate
        self.v_ego = v_ego
        self.v_front = v_front
        self.distance = distance

        self.total_distance += self.distance
        reward = -(self.total_distance / 10000) - (terminated * 1000)

        return nstate, reward, terminated, False, self._get_info()

    def _step(self, state, action):
        v_ego, v_front, distance = state

        old_vel = v_front - v_ego

        front_acc = self.get_front_acceleration(state)

        if distance <= self.max_sensor_distance:
            front_change = self.speed_limit(
                self.v_front_min,
                self.v_front_max,
                v_front,
                front_acc
            )
            v_front += front_change
        else:
            if front_acc < v_ego:
                distance = 200
                # maybe other stuff here as well
            else:
                v_front = 0

        ego_change = self.speed_limit(
            self.v_ego_min,
            self.v_ego_max,
            v_ego,
            self.get_ego_acceleration(action)
        )
        v_ego += ego_change

        new_vel = v_front - v_ego
        distance += (old_vel + new_vel) / 2
        distance = min(distance, self.max_sensor_distance + 1)

        terminated = distance <= 0

        # this is just to return something, but it is overwritten in step()
        reward = 0

        return (v_ego, v_front, distance), reward, terminated


    def speed_limit(self, vmin, vmax, vcurr, acc):
        # if either already driving to slow or to fast, return neutral
        if (acc == self.DEC and vcurr <= vmin) or \
                (acc == self.ACC and vcurr >= vmax):
            return self.NEU

        # else return intended acceleration
        else:
            return acc

    def get_ego_acceleration(self, action):
        return self.action_to_acceleration[action]

    def get_front_acceleration(self, state):
        # it's unlucky if the front car are always backing
        if self.unlucky:
            return self.DEC

        # otherwise just sample a random action
        v_ego, v_front, distance = state
        if distance <= self.max_sensor_distance:
            return np.random.choice(self.action_to_acceleration)

        # this is from Asgers implementation, but I don't really get it..
        elif np.random.random() < 0.5 and v_ego > self.v_front_min:
            return np.random.choice(np.arange(self.v_front_min - 1, v_ego))
        else:
            return self.v_front_max

        pass


class DCDCBoostConverterEnv(gym.Env):
    metadata = { 'render_mode': [] }

    # hybrid states
    D1U1 = 0
    D1U0 = 1
    D0U0 = 2
    PU0D0 = 3
    PD1U1 = 4
    PD1U0 = 5

    def __init__(
        self, render_mode=None,
        obs_low=[0, 14.4, 30.], obs_high=[4.1, 15.6, 200.], unlucky=False
    ):
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(2)

        # state variables
        self.x1 = 0.
        self.x2 = 0.
        self.R = 73.

        # hybrid state
        self.hybrid_state = self.PU0D0

        # target values
        self.x1_ref = 0.35
        self.x2_ref = 15.

        # minimize this (distance from x1,x2 to target values)
        self.dist = 0

        # safety constraints
        self.x1_min = 0
        self.x1_max = 4
        self.x2_min = 14.5
        self.x2_max = 15.5

        # constants
        self.L = 450e-6             # Inductors inductance
        self.RL = 0.3               # Inductors resistance
        self.Co = 220e-6            # Capacitators capacitance
        self.vs = 10.0              # Input voltage
        self.scale = 1.0 / 0.00005  # time scale

        self.R_fluct= 30.  # Output consumption
        self.R_min = 30.
        self.R_max = 200.

        # euler method step size
        self.h = 0.1

    def is_safe(self, state):
        x1, x2, _ = state
        return self.x1_min <= x1 <= self.x1_max and \
            self.x2_min <= x2 <= self.x2_max

    def is_terminal(self, state):
        return False if self.is_safe(state) else True

    def allowed_actions(self, state):
        return [0,1] if self.is_safe(state) else []

    def reset(self):
        self.dist = 0.
        self.x1 = 0.
        self.x2 = 0.
        self.R = 73.  # Output consumption
        self.hybrid_state = self.PU0D0

        return self._get_obs(), self._get_info()

    def step(self, action):
        nstate = self._step(self._get_obs(), action)
        reward = self.reward(nstate)
        terminated = self.is_terminal(nstate)

        x1, x2, R = nstate
        self.x1 = x1
        self.x2 = x2
        self.R = R
        self.dist += self.get_dist(nstate)

        return nstate, reward, terminated, False, self._get_info()

    def step_from(self, state, action):
        nstate = self._step(state, action)
        return nstate, self.reward(nstate), self.is_terminal(nstate)

    def _step(self, state, action):
        x1, x2, R = state

        R_update = self._get_random_R()
        while R + R_update <= self.R_min or R + R_update >= self.R_max:
            R_update = self._get_random_R()
        R += R_update

        L, RL, Co, vs, s = self._get_constants()

        for _ in range(int(1 / self.h)):
            # on
            if action == 1:
                self.hybrid_state = self.PD1U1
                x1 += self.h * (-RL * x1 / L + vs / L) / s
                x2 += self.h * (-x2 / Co * R) / s

            # off
            elif action == 0 and x1 > 0:
                self.hybrid_state = self.PD1U0
                x1 += self.h * (-RL * x1 / L + -x2 / L + vs / L) / s
                x2 += self.h * (x1 / Co + -x2 / Co * R) / s

            # off but x1 is negative
            elif action == 0 and x1 <= 0:
                self.hybrid_state = self.PU0D0
                x1 += 0
                x2 += self.h * (-x2 / Co * R) / s

        return (x1, x2, R)

    def get_dist(self, state):
        x1, x2, R = state
        r1 = abs(x1 - self.x1_ref)
        r2 = (x2 - self.x2_ref)**2
        return r2 + r1 / 3.0

    def reward(self, state):
        return -self.get_dist(state)

    def _get_obs(self):
        return [self.x1, self.x2, self.R]

    def _get_info(self):
        return { 'hybrid_state': self.hybrid_state }

    def _get_random_R(self):
        return np.random.randint(-self.R_fluct, self.R_fluct + 1)

    def _get_constants(self):
        return self.L, self.RL, self.Co, self.vs, self.scale


class OilPumpEnv(gym.Env):
    metadata = { 'render_modes': [] }

    OFF = 0
    ON = 1

    def __init__(
        self, render_mode=None,
        obs_low=[0, 4.9, 0, 0], obs_high=[20, 25.1, 1, 2], unlucky=False
    ):

        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32)
        )
        self.action_space = gym.spaces.Discrete(2)
        self.unlucky = unlucky

        # mechanics
        self.v_min = 4.9
        self.v_max = 25.1
        self.period = 20
        self.time_step = 0.2
        self.inflow = 0.2
        self.fluctuation = 0.1
        self.imprecision = 0.06
        self.latency = 2

        # track the time that has elapsed and total accumulated oil volume
        self.elapsed = 0.
        self.accumulated_v = 0.

        # state is tuple (t, v, p, l) where
            # t is  the time in the consumption cycle
            # v is the volume of oil in the tank
            # p is the pump status ('on' or 'off')
            # l is th latency timer controlling how often the pump can switch
            # state

        self.reset()

    def reset(self):
        self.t = 0.
        self.v = 10.
        self.p = 0.
        self.l = 0.

        self.elapsed = 0.
        self.accumulated_v = 0.

        return self._get_obs(), self._get_info()

    def is_safe(self, state):
        _, v, _, _ = state
        return self.v_min <= v <= self.v_max

    def allowed_actions(self, state):
        return [0, 1] if self.is_safe(state) else []

    def reward(self, state):
        elapsed = self.elapsed if self.elapsed > 0 else self.time_step
        return (self.time_step / elapsed) * self.accumulated_v

    def step(self, action):
        nstate = self._step(self._get_obs(), action)

        t, v, p, l = nstate
        self.t = t
        self.v = v
        self.p = p
        self.l = l

        self.accumulated_v += v
        self.elapsed += self.time_step

        reward = self.reward(nstate)
        terminated = not self.is_safe(nstate)

        return nstate, reward, terminated, False, self._get_obs()

    def step_from(self, state, action):
        nstate = self._step(state, action)
        return nstate, self.reward(nstate), not self.is_safe(nstate)

    def _step(self, state, action):
        t, v, p, l = state
        p = round(p)

        if action != p and l <= 0.0:
            p = action
            l = self.latency

        td = t
        fluctuation_updated = False
        fluctuation = self._get_random()
        while td < t + self.time_step:

            # update fluctuation if we are over halway there
            if td >= t + self.time_step / 2:
                if not fluctuation_updated:
                    fluctuation = self._get_random()
                    fluctuation_updated = True

            time_step = min(
                t + self.time_step - td,
                self.next_rate_change(td) - td
            )
            consumption = self.consumption_rate(td)
            if consumption > 0:
                consumption += fluctuation

            v = v - consumption * time_step
            td += time_step

        td = td % 20
        l -= self.time_step
        if p == self.ON:
            v += self.inflow * self.time_step

        return td, v, p, l

    def next_rate_change(self, t):
        return t - (t % 2) + 2

    def consumption_rate(self, t):
        if t < 2:
            return 0
        if t < 4:
            return 1.2
        if t < 8:
            return 0
        if t < 10:
            return 2.5
        if t < 12:
            return 2.5
        if t < 14:
            return 0
        if t < 16:
            return 1.7
        if t < 18:
            return 0.5
        return 0

    def _get_random(self):
        return np.random.random() * (self.fluctuation * 2) - self.fluctuation

    def _get_obs(self):
        return np.array([self.t, self.v, self.p, self.l], dtype=np.float32)

    def _get_info(self):
        return {}
