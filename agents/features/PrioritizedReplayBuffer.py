import numpy as np
import agents.features.ReplayBuffer as ReplayBuffer


class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (proportional variant).
    Equations follow Schaul et al. 2016:
       p_i = (|delta_i| + eps)^alpha
       P(i) = p_i / sum_k p_k
       w_i = (1 / (N * P(i)))^beta  / max_j w_j
    """
      
    def __init__(self, alpha=0.5, beta0=0.4, beta_steps=1_000_000, eps=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.beta0 = beta0
        self.beta_steps = beta_steps
        self.eps = eps

        # priorities (initialize to 1.0 so new items are sampled)
        self.priorities = np.ones(self.mem_size, dtype=np.float32)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

    def store_transition(self, state, action, reward, next_state, done, init_priority=None):
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx]      = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx]     = int(action)
        self.reward_memory[idx]     = reward
        self.terminal_memory[idx]   = bool(done)

        # if no priority provided, use current max to ensure the new sample is seen at
        # least once.
        if init_priority is None:
            max_p = self.priorities[:len(self)].max() if len(self) > 0 else 1.0
            self.priorities[idx] = max_p
        else:
            # convert to p_i = (|delta| + eps)^alpha if caller passes delta
            self.priorities[idx] = (abs(float(init_priority)) + self.eps) ** self.alpha

        self.mem_cntr += 1

    def _beta_at(self, step):
        """Linear anneal beta from beta0 → 1.0 over beta_steps."""
        if self.beta_steps <= 0:
            return 1.0
        frac = min(1.0, max(0.0, step / float(self.beta_steps)))
        return self.beta0 + (1.0 - self.beta0) * frac #gradually increase proportional to # learning steps

    def sample_from_buffer(self, batch_size, step=0):
        """
        Returns:
           states, actions, rewards, next_states, terminals, indices, is_weights
        """
        size = len(self)
        if size == 0:
            raise ValueError("PER buffer is empty.")
        batch_size = min(batch_size, size)

        # probabilities P(i) ∝ p_i
        p = self.priorities[:size]
        if self.alpha == 0.0:
            P = np.full_like(p, 1.0 / size)  # uniform sampling
        else:
            P = p / p.sum()

        idxs = np.random.choice(size, batch_size, replace=False, p=P)

        # importance-sampling weights
        beta = self._beta_at(step)
        w = (size * P[idxs]) ** (-beta)
        w = w / w.max()  # normalize for stability
        w = w.astype(np.float32)

        states      = self.state_memory[idxs]
        actions     = self.action_memory[idxs]
        rewards     = self.reward_memory[idxs]
        next_states = self.next_state_memory[idxs]
        terminals   = self.terminal_memory[idxs]

        return states, actions, rewards, next_states, terminals, idxs, w

    def update_priorities(self, idxs, td_errors):
        """
        Update priorities after a learning step.
        td_errors: numpy array or list, shape [batch]
        """
        td_errors = np.asarray(td_errors, dtype=np.float32)
        # p_i = (|delta| + eps)^alpha
        new_p = (np.abs(td_errors) + self.eps) ** self.alpha
        self.priorities[idxs] = new_p