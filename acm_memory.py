from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
from rl.memory import SequentialMemory, RingBuffer
import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action0, reward0, state1, action1, reward1,state2, terminal1, terminal2')


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        # We have enough data. Draw without replacement, that is each index is unique in the
        # batch. We cannot use `np.random.choice` here because it is horribly inefficient as
        # the memory grows. See https://github.com/numpy/numpy/issues/2764 for a discussion.
        # `random.sample` does the same thing (drawing without replacement) and is way faster.
        try:
            r = xrange(low, high)
        except NameError:
            r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        # Not enough data. Help ourselves with sampling from the range, but the same index
        # can occur multiple times. This is not good and should be avoided by picking a
        # large enough warm-up phase.
        warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs

class Acm_Memory(SequentialMemory):
    def __init__(self, limit, **kwargs):
      super(Acm_Memory, self).__init__(**kwargs)

      self.limit = limit

      # Do not use deque to implement the memory. This data structure may seem convenient but
      # it is way too slow on random access. Instead, we use our own ring buffer implementation.
      self.actions = RingBuffer(limit)
      self.rewards = RingBuffer(limit)
      self.terminals = RingBuffer(limit)
      self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            # Draw random indexes such that we have at least a single entry before each
            # index.
            batch_idxs = sample_batch_indexes(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip this transition because the environment was reset here. Select a new, random
                # transition and use this instead. This may cause the batch to contain the same
                # transition twice.
                idx = sample_batch_indexes(1, self.nb_entries, size=1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # This code is slightly complicated by the fact that subsequent observations might be
            # from different episodes. We ensure that an experience never spans multiple episodes.
            # This is probably not that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 2]]
            for offset in range(0, self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previously handled observation was terminal, don't add the current one.
                    # Otherwise we would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            while len(state0) < self.window_length:
                state0.insert(0, zeroed_observation(state0[0]))
            action0 = self.actions[idx - 2]
            reward0 = self.rewards[idx - 2]
            terminal1 = self.terminals[idx - 2]
            action1 = self.actions[idx - 1]
            reward1 = self.rewards[idx - 1]
            state1 = self.rewards[idx - 1]
            terminal2 = self.terminals[idx - 1]
            # Okay, now we need to create the follow-up state. This is state0 shifted on timestep
            # to the right. Again, we need to be careful to not include an observation from the next
            # episode if the last state is terminal.
            state2 = [np.copy(x) for x in state0[1:]]
            state2.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(Experience(state0=state0, action0=action0, reward0=reward0,
                                          state1=state1, action1=aciton1, reward1=reward1,
                                          state2=state2, terminal1=terminal1, terminal2=terminal2))
        assert len(experiences) == batch_size
        return experiences
