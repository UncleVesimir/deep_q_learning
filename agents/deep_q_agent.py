import numpy as np
from datetime import datetime
from utils import sanitize_file_string

from networks.DeepQNetwork import DeepQNetwork
from agents.common.ReplayBufferAgent import ReplayBufferAgent




class DeepQAgent(ReplayBufferAgent):
    """
    Deep Q Learning Agent, based on https://arxiv.org/pdf/1312.5602
    """
    def __init__(self, model_name="DQN", *args, **kwargs):
        super().__init__(*args, **kwargs)

        
        self.checkpoint_dir = f"models/{model_name}/{sanitize_file_string(self.env_name)}"
        self.filename_root = f"{model_name}_{sanitize_file_string(self.env_name)}_lr{self.learning_rate}_gamma{self.gamma}_eps{self.epsilon}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.q_eval = DeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_eval", 
            lr=self.learning_rate,
            checkpoint_dir=self.checkpoint_dir
        )

        self.q_target = DeepQNetwork( 
            self.n_actions, 
            self.input_dims, 
            self.filename_root + "_target", 
            lr=self.learning_rate,
            checkpoint_dir=self.checkpoint_dir
        )

        self.networks = [self.q_eval, self.q_target] # for savings model dicts

    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.min_epsilon else self.min_epsilon
    

    def check_if_target_network_needs_replace(self):
        if self.learn_step_cnt % self.replace_limit == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

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