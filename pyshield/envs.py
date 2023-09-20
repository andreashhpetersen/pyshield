import math
import numpy as np
import gymnasium as gym


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
    SLOW = 0
    FAST = 1

    def __init__(self, render_mode='human', obs_low=[0., 1.], obs_high=[0., 1.], unlucky=False):
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
        pass

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
