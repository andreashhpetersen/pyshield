import os
import sys
import resource
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict


class Shield:
    def __init__(self, env, granularity, samples_per_axis=2):
        self.env = env
        self.grid = Grid(env.observation_space, env.action_space, granularity)
        self.safe_actions = np.ones(
            self.grid.shape + (self.env.action_space.n,),
            dtype=bool
        )
        self.reachability_dict = defaultdict(set)
        self.samples_per_axis = samples_per_axis
        self.n_partitions = np.prod(self.grid.shape)

    def find_reachable(self, partition):
        region = self.grid.ridx_to_region(partition)
        sp = self.grid.get_supporting_points(
            region,
            per_axis=self.samples_per_axis
        )

        out = defaultdict(set)
        safe = True
        for s in sp:
            # mark unsafe partitions while we are at it
            if not self.env.is_safe(s):
                safe = False

            # map (action, state) to set of reachable partitions
            for a in self.env.allowed_actions(s):
                ns, _, _ = self.env.step_from(s, a)
                npartition = self.grid.s_to_ridx(ns)
                out[a].add(npartition)

        return (partition, out, safe)

    def compute_reachability_function_single(self):
        data = map(self.find_reachable, self.grid.regions())
        for partition, reachable, safe in tqdm(data, total=self.n_partitions):
            self.reachability_dict[partition] = reachable
            if not safe:
                self.safe_actions[partition] = False

    def compute_reachability_function_multi(self):
        with Pool() as p:
            res = p.map(self.find_reachable, self.grid.regions())
            for partition, reachable, safe in res:
                self.reachability_dict[partition] = reachable
                if not safe:
                    self.safe_actions[partition] = False

    def make_shield(self, max_steps=1000, multi=True, verbosity=0):
        if verbosity > 0:
            print('computing reachability function...')

        try:
            if multi:
                self.compute_reachability_function_multi()
            else:
                self.compute_reachability_function_single()
        except KeyboardInterrupt:
            print('KeyboardInterrupt, exit quietly')
            return

        # iteratively synthesize shield until fixed point (or max iterations)
        print('synthesizing shield...')
        for i in range(max_steps):

            updates = self._synthesize()
            if updates == 0:
                print(f'finished early after {i+1} iterations')
                break

            elif verbosity > 0:
                print(f'num updates at iteration {i+1}: {updates}')

                if verbosity > 1 and i % 5 == 0:
                    self.draw(actions=['nohit', 'hit'], out_fp=f'./shield_{i+1}.png')

    def _synthesize(self):
        # keep track of how many updates we make (to check for fixed point)
        updates = 0
        for partition in self.grid.regions():

            # get safe actions at this partition
            safe_actions = np.argwhere(self.safe_actions[partition]).T[0]
            for a in safe_actions:

                # get reachable partitions
                reachable = self.reachable(partition, a)
                for rpartition in reachable:
                    if not self.safe_actions[rpartition].any():
                        self.safe_actions[partition][a] = False
                        updates += 1
                        break

        return updates

    def reachable(self, bin_id, a):
        return self.reachability_dict[bin_id][a]

    def draw(self, cmap, axis_labels=('x','y'),
             actions=None, out_fp='./shield.png', show=False, lw=0):
        fig, ax = plt.subplots(figsize=(8,6))

        labels = []
        # cmap = { '()': 'r', '(hit)': 'g', '(nohit)': 'y', '(nohit, hit)': 'w' }

        for ridx in self.grid.regions():
            region = self.grid.ridx_to_region(ridx)
            safe_actions = np.argwhere(self.safe_actions[ridx]).T[0]
            safe_actions = [actions[a] for a in safe_actions]

            label = '({})'.format(', '.join(safe_actions))
            labels.append(label)

            x_start, x_end = region[0]
            y_start, y_end = region[1]

            width = x_end - x_start
            height = y_end - y_start
            c = cmap[label]

            ec = 'k' if lw > 0 else None
            ax.add_patch(
                mpatches.Rectangle(
                    (x_start, y_start), width, height, color=c,
                    lw=lw, ec=ec, zorder=0
                )
            )

        proxies = []
        for label in set(labels):
            proxies.append(mpatches.Patch(label=label, color=cmap[label]))

        plt.legend(handles=proxies, loc='lower right')

        xl, yl = self.env.observation_space.low
        xh, yh = self.env.observation_space.high

        plt.xlim([xl, xh])
        plt.ylim([yl, yh])

        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

        plt.tight_layout()

        if show:
            plt.show()

        if out_fp is not None:
            plt.savefig(out_fp)

        plt.close()


class Grid:
    def __init__(self, observation_space, action_space, granularity):
        self.obs_space = observation_space
        self.act_space = action_space
        self.g = granularity

        assert len(self.obs_space.shape) == 1

        self.n_features = self.obs_space.shape[0]
        self._shape = np.array(
            (self.obs_space.high - self.obs_space.low) // self.g,
            dtype=np.int16
        )
        self.dtype = self.obs_space.dtype

    @property
    def shape(self):
        return tuple(self._shape)

    def s_to_ridx(self, s):
        s = np.array(s, dtype=self.dtype)

        res = np.array((s - self.obs_space.low) // self.g, dtype=np.int16)
        res = np.vstack((res, self._shape-1)).T.min(axis=1)
        res[res < 0] = 0

        return tuple(res)

    def ridx_to_region(self, ridx):
        if not isinstance(ridx, np.ndarray):
            ridx = np.array(ridx, dtype=np.int16)

        low = (ridx * self.g) + self.obs_space.low
        high = low + self.g
        return np.vstack((low, high), dtype=self.dtype).T

    def get_supporting_points(self, region, per_axis=2, eps=1e-6):
        coordinates = np.linspace(region[:,0], region[:,1] - eps, num=per_axis)
        return np.array(list(itertools.product(*coordinates.T)))

    def regions(self, idx=True):
        steps = [np.arange(n) for n in self.shape]
        for ridx in itertools.product(*steps):
            yield ridx if idx else self.ridx_to_region(ridx)
