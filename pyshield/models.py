import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

    def compute_reachability_function(self):
        for partition in self.grid.bins():
            region = self.grid.bins_to_region(partition)
            sp = self.grid.get_supporting_points(
                region,
                per_axis=self.samples_per_axis
            )

            for s in sp:
                # mark unsafe partitions while we are at it
                if not self.env.is_safe(s):
                    self.safe_actions[partition] = False

                # map (action, state) to set of reacable partitions
                for a in self.env.allowed_actions(s):
                # for a in range(self.env.action_space.n):
                    ns, _, _ = self.env.step_from(s, a)
                    npartition = self.grid.s_to_bin(ns)
                    self.reachability_dict[(a,) + partition].add(npartition)

    def make_shield(self, max_steps=1000, verbosity=0):
        if verbosity > 0:
            print('computing reachability function...')

        self.compute_reachability_function()

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
        for partition in self.grid.bins():

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
        return self.reachability_dict[(a,) + bin_id]

    def draw(self, cmap, axis_labels=('x','y'),
             actions=None, out_fp='./shield.png', show=False, lw=0):
        fig, ax = plt.subplots(figsize=(8,6))

        labels = []
        # cmap = { '()': 'r', '(hit)': 'g', '(nohit)': 'y', '(nohit, hit)': 'w' }

        for bin_ids in self.grid.bins():
            region = self.grid.bins_to_region(bin_ids)
            safe_actions = np.argwhere(self.safe_actions[bin_ids]).T[0]
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
        self.n_bins = np.array(
            (self.obs_space.high - self.obs_space.low) // self.g,
            dtype=np.int16
        )
        self.dtype = self.obs_space.dtype

    @property
    def shape(self):
        return tuple(self.n_bins)

    def s_to_bin(self, s):
        s = np.array(s, dtype=self.dtype)

        # try:
        #     assert self.obs_space.contains(s)
        # except AssertionError:
        #     s = np.vstack((s, self.obs_space.low + 1e-6)).T.max(axis=1)
        #     s = np.vstack((s, self.obs_space.high - 1e-6)).T.min(axis=1)

        res = np.array((s - self.obs_space.low) // self.g, dtype=np.int16)
        res = np.vstack((res, self.n_bins-1)).T.min(axis=1)

        return tuple(res)

    def bins_to_region(self, bins):
        if not isinstance(bins, np.ndarray):
            bins = np.array(bins, dtype=np.int16)

        low = (bins * self.g) + self.obs_space.low
        high = low + self.g
        return np.vstack((low, high), dtype=self.dtype).T

    def sample_from_bins(self, bins, n=1):
        region = self.bins_to_region(bins)
        return self.sample_from_region(region, n=n)

    def sample_from_region(self, region, n=1):
        return np.random.uniform(
            region[:,0],
            region[:,1],
            size=(n, self.n_features)
        )

    def get_supporting_points(self, region, per_axis=2, eps=1e-6):
        coordinates = np.linspace(region[:,0], region[:,1] - eps, num=per_axis)
        return np.array(list(itertools.product(*coordinates.T)))

    def bins(self):
        steps = [np.arange(n_bins) for n_bins in self.n_bins]
        for bin_ids in itertools.product(*steps):
            yield bin_ids
