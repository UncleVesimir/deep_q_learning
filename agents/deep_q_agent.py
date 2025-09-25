from collections import defaultdict
import torch
import numpy as np

from networks.deepQNetwork import DeepQNetwork


class DeepQAgent():
    def __init__(self, n_actions, input_dims, gamma=0.99, epsilon=1.0, epsilon_dec=5e-7, min_epsilon=0.01,
                  batch_size=32, learning_rate=0.1, replace_limit=1000, env_name=None):
             # Q(s,a)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.learn_step_cnt = 0
        self.replace_limit = replace_limit
        self.batch_size = batch_size
        self.env_name=env_name

        safe_env = (env_name or "env").replace("/", "_").replace("\\", "_")
        self.filename_root = f"DQN_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}__{safe_env}"

        self.memory = ReplayBuffer(mem_size=100_000, input_shape=input_dims, n_actions=n_actions)


        self.q_eval = DeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_eval", 
            lr=self.learning_rate
        )

        self.q_target = DeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_target", 
            lr=self.learning_rate
        )

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.q_eval.device) # unsqueeze to add required batch dimension
            action = self.q_eval(state).argmax().item()
        else:
            action = np.random.choice(self.n_actions)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.min_epsilon else self.min_epsilon
    

    def check_if_target_network_needs_replace(self):
        if self.learn_step_cnt % self.replace_limit == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

    def store_transition(self, state, action, reward, next_state, terminal):
        self.memory.store_transition(state, action, reward, next_state, terminal)

    
    def sample_from_buffer(self):
        states, actions, rewards, next_states, terminals = self.memory.sample_from_buffer(self.batch_size)

        states = torch.tensor(states).to(self.q_eval.device)
        actions = torch.tensor(actions).to(self.q_eval.device)
        rewards = torch.tensor(rewards).to(self.q_eval.device)
        next_states = torch.tensor(next_states).to(self.q_eval.device)
        terminals = torch.tensor(terminals).to(self.q_eval.device)

        return states, actions, rewards, next_states, terminals

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: ## skip until we have batch_size in memory buffer (from main.py env loop)
            return
        self.q_eval.optimizer.zero_grad()

        self.check_if_target_network_needs_replace() ## check if we need to update target network weights to eval weights

        states, actions, rewards, next_states, terminals = self.sample_from_buffer()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval(states)[indices, actions]
        q_next_state = self.q_target(next_states).max(dim=1)[0]
        
        q_next_state[terminals] = 0.0

        q_targ = rewards + self.gamma *  q_next_state

        loss = self.q_eval.loss(q_pred, q_targ).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_cnt += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_target.load_checkpoint()


## TODO: move to separate file as I'll need to use these in other Agents
class ReplayBuffer():
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_from_buffer(self, batch_size):
        size = min(self.mem_cntr, self.mem_size) ## avoid sampling from uninitialized memory
        batch_size = min(batch_size, size) ## ensure we don't sample more than we have

        batch = np.random.choice(size, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals