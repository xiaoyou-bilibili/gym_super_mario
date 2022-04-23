import numpy as np


class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self._alpha = alpha
        self._capacity = capacity
        self._buffer = []
        self._position = 0
        self._priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self._priorities.max() if self._buffer else 1.0

        batch = (state, action, reward, next_state, done)
        if len(self._buffer) < self._capacity:
            self._buffer.append(batch)
        else:
            self._buffer[self._position] = batch

        self._priorities[self._position] = max_prio
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size, beta=0.4):
        if len(self._buffer) == self._capacity:
            prios = self._priorities
        else:
            prios = self._priorities[:self._position]

        probs = prios ** self._alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self._buffer), batch_size, p=probs)
        samples = [self._buffer[idx] for idx in indices]

        total = len(self._buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self._priorities[idx] = prio

    def __len__(self):
        return len(self._buffer)
