import dataclasses
from typing import Any

import jax
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze

def normalize(scale, data):
    if scale is None: return data
    b = (scale['max'] + scale['min']) / 2
    s = (scale['max'] - scale['min']) / 2
    return (data - b) / (s + 1e-3)

def unnormalize(scale, data):
    if scale is None: return data
    b = (scale['max'] + scale['min']) / 2
    s = (scale['max'] - scale['min']) / 2
    return data * (s + 1e-3) + b

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, scale=None, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        #print(data.keys())
        if scale is not None:
          for scale_key in scale:
            for key in data:
              if scale_key in key:
                data[key] = normalize(scale[scale_key], data[key])
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

        #(self.terminal_locs,) = np.nonzero(self['terminals'] > 0)
        #self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def add_batch(self, transitions):
        """Add a transition to the replay buffer."""
        left = self.max_size - self.pointer
        N = transitions['observations'].shape[0]
        if N > left:
            add_batch(self, {k: v[:left] for k, v in transitions.items()})
            add_batch(self, {k: v[left:] for k, v in transitions.items()})
            return

        _dict = unfreeze(self._dict)
        for k, v in transitions.items():
            _dict[k] = np.array(_dict[k], copy=True)  
            _dict[k][self.pointer: self.pointer + N] = v
        self._dict = freeze(_dict)
        self.pointer = (self.pointer + N) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
            batch['actor_goals'] = self.dataset['oracle_reps'][actor_goal_idxs]
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
            batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])


@dataclasses.dataclass
class HGCDataset(GCDataset):
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support hierarchical goal-conditioned RL. It reads the following additional key from
    the config:
    - subgoal_steps (optional: value_subgoal_steps, actor_subgoal_steps): Subgoal steps. It is also possible to specify
        `value_subgoal_steps` and `actor_subgoal_steps` separately.
    - low_discount: If specified, return low-level value goals as well.
    """

    def compute_high_next_idxs(self, idxs, final_state_idxs, high_goal_idxs, subgoal_steps):
        """Compute the next indices for high-level goals."""
        batch_size = len(idxs)
        subgoal_steps = np.full(batch_size, subgoal_steps)

        # Clip to the end of the trajectory.
        subgoal_steps = np.minimum(subgoal_steps, final_state_idxs - idxs)

        # Clip to the high-level goal.
        diff_idxs = high_goal_idxs - idxs
        should_clip = (0 <= diff_idxs) & (diff_idxs < subgoal_steps)
        subgoal_steps = np.where(should_clip, diff_idxs, subgoal_steps)

        return idxs + subgoal_steps, subgoal_steps

    def get_high_actions(self, target_idxs, cur_idxs):
        if 'oracle_reps' in self.dataset:
            return self.dataset['oracle_reps'][target_idxs]
        else:
            return self.get_observations(target_idxs)

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]

        # Sample high-level value goals.
        high_value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        value_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('value_subgoal_steps') is None
            else self.config['value_subgoal_steps']
        )
        high_value_next_idxs, high_value_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_value_goal_idxs,
            value_subgoal_steps,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_value_reps'] = self.dataset['oracle_reps'][idxs]
            batch['high_value_goals'] = self.dataset['oracle_reps'][high_value_goal_idxs]
            batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        else:
            batch['high_value_reps'] = batch['observations']
            batch['high_value_goals'] = self.get_observations(high_value_goal_idxs)
            batch['high_value_actions'] = self.get_high_actions(high_value_next_idxs, idxs)
            batch['high_value_next_observations'] = self.get_observations(high_value_next_idxs)
        batch['high_value_offsets'] = high_value_goal_idxs - idxs

        high_value_successes = (high_value_subgoal_steps < value_subgoal_steps).astype(float)
        batch['high_value_subgoal_steps'] = high_value_subgoal_steps
        batch['high_value_masks'] = 1.0 - high_value_successes
        if self.config['gc_negative']:
            batch['high_value_rewards'] = -(1 - self.config['discount'] ** high_value_subgoal_steps) / (
                1 - self.config['discount']
            )
        else:
            batch['high_value_rewards'] = (self.config['discount'] ** high_value_subgoal_steps) * high_value_successes

        # Sample low-level value goals (if requested).
        if 'low_discount' in self.config:
            low_value_goal_idxs = self.sample_goals(
                idxs,
                self.config['value_p_curgoal'],
                self.config['value_p_trajgoal'],
                self.config['value_p_randomgoal'],
                geom_sample=True,
                discount=self.config['low_discount'],
            )

            if 'oracle_reps' in self.dataset:
                batch['low_value_goals'] = self.dataset['oracle_reps'][low_value_goal_idxs]
            else:
                batch['low_value_goals'] = self.get_observations(low_value_goal_idxs)
            successes = (idxs == low_value_goal_idxs).astype(float)
            batch['low_value_masks'] = 1.0 - successes
            batch['low_value_rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # One-step information.
        successes = (idxs == high_value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)

        # Sample high-level actor goals.
        high_actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )
        actor_subgoal_steps = (
            self.config['subgoal_steps']
            if self.config.get('actor_subgoal_steps') is None
            else self.config['actor_subgoal_steps']
        )
        high_actor_next_idxs, high_actor_subgoal_steps = self.compute_high_next_idxs(
            idxs,
            final_state_idxs,
            high_actor_goal_idxs,
            actor_subgoal_steps,
        )

        if 'oracle_reps' in self.dataset:
            batch['high_actor_goals'] = self.dataset['oracle_reps'][high_actor_goal_idxs]
            batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
            batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)
        else:
            batch['high_actor_goals'] = self.get_observations(high_actor_goal_idxs)
            batch['high_actor_actions'] = self.get_high_actions(high_actor_next_idxs, idxs)
            batch['high_actor_next_observations'] = self.get_observations(high_actor_next_idxs)

        # Compute low-level actor goals.
        low_actor_goal_idxs = np.minimum(idxs + actor_subgoal_steps, final_state_idxs)

        batch['low_actor_goals'] = self.get_high_actions(low_actor_goal_idxs, idxs)

        return batch

@dataclasses.dataclass
class ACGCDataset:
    """Dataset class for goal-conditioned RL with action chunking.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.

    It reads the following keys from the config:
    - action_chunking: Size of the chunk for action chunking.
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        self.chunk_next_obs = self.config['chunk_next_obs'] if 'chunk_next_obs' in self.config else False
        self.use_return_to_go = self.config['return_to_go'] if 'return_to_go' in self.config else False

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        valid_idxs = []
        for i in range(len(self.initial_locs)):
            valid_idxs.append(np.arange(self.initial_locs[i], self.terminal_locs[i] - self.config['action_chunking']))
        self.valid_idxs = np.concatenate(valid_idxs)
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config['value_p_curgoal'] + self.config['value_p_trajgoal'] + self.config['value_p_randomgoal'], 1.0
        )
        assert np.isclose(
            self.config['actor_p_curgoal'] + self.config['actor_p_trajgoal'] + self.config['actor_p_randomgoal'], 1.0
        )

    def sample_consecutive(self, batch_size: int, batch_length: int):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        valid_idxs = self.valid_idxs[self.valid_idxs + self.config['action_chunking'] * batch_length < self.terminal_locs[-1]]
        x = self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs)]
        y = self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs + self.config['action_chunking'] * batch_length)]
        new_valid_idxs = valid_idxs[x==y]

        idxs = np.random.choice(new_valid_idxs, size=batch_size)

        batch_idxs = np.stack([idxs+i for i in range(batch_length * self.config['action_chunking'])], axis=0) 
        _batch = self.dataset.sample(0, batch_idxs)
        _batch = {
            k: np.swapaxes(v.reshape(batch_size, batch_length, self.config['action_chunking'], *v.shape[2:]), 0, 1)
            for k,v in _batch.items()
        }

        batch = {
            'observations': _batch['observations'][:, :, 0],
            'actions': np.swapaxes(np.concatenate(np.swapaxes(_batch['actions'], 0, 2), axis=-1), 0, 1),
        }
        if batch['observations'].dtype == np.uint8:
            batch['observations'] = batch['observations'].astype(np.float32) / 255.0

        return batch



    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = np.random.choice(self.valid_idxs, size=batch_size)

        ac_actor = self.config['action_chunking_actor'] if 'action_chunking_actor' in self.config and self.config['action_chunking_actor'] > 0 else self.config['action_chunking']
        batch_idxs = np.stack([idxs+i for i in range(self.config['action_chunking']+1)], axis=0) 
        _batch = self.dataset.sample(0, batch_idxs)

        batch = {
            'observations': _batch['observations'][0],
            'actions': np.concatenate(_batch['actions'][:ac_actor], axis=-1), 
            'next_observations': np.concatenate(_batch['observations'][1:], axis=-1) if self.chunk_next_obs else _batch['observations'][-1],
            'next_goals': _batch['oracle_reps'][-1],
        }

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config['value_p_curgoal'],
            self.config['value_p_trajgoal'],
            self.config['value_p_randomgoal'],
            self.config['value_geom_sample'],
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config['actor_p_curgoal'],
            self.config['actor_p_trajgoal'],
            self.config['actor_p_randomgoal'],
            self.config['actor_geom_sample'],
        )

        if 'oracle_reps' in self.dataset:
            batch['value_goals'] = self.dataset['oracle_reps'][value_goal_idxs]
            batch['actor_goals'] = self.dataset['oracle_reps'][actor_goal_idxs]
        else:
            batch['value_goals'] = self.get_observations(value_goal_idxs)
            batch['actor_goals'] = self.get_observations(actor_goal_idxs)

        successes = np.logical_and(idxs <= value_goal_idxs, value_goal_idxs < idxs + self.config['action_chunking']).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config['gc_negative'] else 0.0)
        if batch['observations'].dtype == np.uint8:
            batch['observations'] = batch['observations'].astype(np.float32) / 255.0
        if self.use_return_to_go:
            terminal_locs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
            batch['returns_to_go'] = np.logical_and(idxs <= value_goal_idxs, value_goal_idxs < terminal_locs).astype(float)
            #batch['returns_to_go'] = np.where(np.logical_and(idxs <= value_goal_idxs, value_goal_idxs < terminal_locs), np.pow(self.config['discount'], value_goal_idxs-idxs), jnp.zeros_like(batch['rewards']))
            #print(batch['returns_to_go'].min(), batch['returns_to_go'].max(), batch['returns_to_go'].mean())

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goals for the given indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config['discount']

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )

            # Goals at the current state.
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])

@dataclasses.dataclass
class CDataset:
    """Dataset class without action chunking.
    
    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        valid_idxs = []
        for i in range(len(self.initial_locs)):
            valid_idxs.append(np.arange(self.initial_locs[i], self.terminal_locs[i] - 1))
        self.valid_idxs = np.concatenate(valid_idxs)
        assert self.terminal_locs[-1] == self.size - 1

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = np.random.choice(self.valid_idxs, size=batch_size)

        batch = self.dataset.sample(0, idxs)
        return batch

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])




@dataclasses.dataclass
class ACDataset:
    """Dataset class with action chunking.
    
    It reads the following keys from the config:
    - action_chunking: Size of the chunk for action chunking.
    
    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
    """

    dataset: Dataset
    config: Any

    def __post_init__(self):
        self.size = self.dataset.size
        self.chunk_next_obs = self.config['chunk_next_obs'] if 'chunk_next_obs' in self.config else False
        self.use_return_to_go = self.config['return_to_go'] if 'return_to_go' in self.config else False

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        valid_idxs = []
        for i in range(len(self.initial_locs)):
            valid_idxs.append(np.arange(self.initial_locs[i], self.terminal_locs[i] - self.config['action_chunking']))
        self.valid_idxs = np.concatenate(valid_idxs)

        if self.use_return_to_go:
            S = np.cumsum(self.dataset['rewards'])
            terminal_locs = self.terminal_locs[np.searchsorted(self.terminal_locs, np.arange(self.size))]
            self.return_to_go = S[terminal_locs] - S + self.dataset['rewards']
            #print(self.return_to_go.min(), self.return_to_go.max(), self.return_to_go.mean(), self.dataset['rewards'].min())

        assert self.terminal_locs[-1] == self.size - 1

    def sample_consecutive(self, batch_size: int, batch_length: int):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        valid_idxs = self.valid_idxs[self.valid_idxs + self.config['action_chunking'] * batch_length < self.terminal_locs[-1]]
        x = self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs)]
        y = self.terminal_locs[np.searchsorted(self.terminal_locs, valid_idxs + self.config['action_chunking'] * batch_length)]
        new_valid_idxs = valid_idxs[x==y]

        idxs = np.random.choice(new_valid_idxs, size=batch_size)

        batch_idxs = np.stack([idxs+i for i in range(batch_length * self.config['action_chunking'])], axis=0) 
        _batch = self.dataset.sample(0, batch_idxs)
        _batch = {
            k: np.swapaxes(v.reshape(batch_size, batch_length, self.config['action_chunking'], *v.shape[2:]), 0, 1)
            for k,v in _batch.items()
        }

        batch = {
            'observations': _batch['observations'][:, :, 0],
            'actions': np.swapaxes(np.concatenate(np.swapaxes(_batch['actions'], 0, 2), axis=-1), 0, 1),
        }
        if batch['observations'].dtype == np.uint8:
            batch['observations'] = batch['observations'].astype(np.float32) / 255.0

        return batch



    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation.
        """
        if idxs is None:
            idxs = np.random.choice(self.valid_idxs, size=batch_size)

        ac_actor = self.config['action_chunking_actor'] if 'action_chunking_actor' in self.config and self.config['action_chunking_actor'] > 0 else self.config['action_chunking']
        batch_idxs = np.stack([idxs+i for i in range(self.config['action_chunking']+1)], axis=0) 
        _batch = self.dataset.sample(0, batch_idxs)
        batch = {
            'observations': _batch['observations'][0],
            'actions': np.concatenate(_batch['actions'][:ac_actor], axis=-1),
            'next_observations': np.concatenate(_batch['observations'][1:], axis=-1) if self.chunk_next_obs else _batch['observations'][-1],
            'rewards': np.mean(_batch['rewards'][:-1] * np.pow(self.config['discount'], np.arange(self.config['action_chunking'])[:, None]), axis=0),
            'value_goals': None,
            'actor_goals': None,
        } 
        if self.use_return_to_go:
            batch['returns_to_go'] = self.return_to_go[batch_idxs]
        batch['masks'] = np.ones_like(batch['rewards'])
        if batch['observations'].dtype == np.uint8:
            batch['observations'] = batch['observations'].astype(np.float32) / 255.0

        return batch

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])



