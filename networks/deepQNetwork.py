import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, file_name, lr=0.0001, checkpoint_dir="models/Unknown"):
        super().__init__()
        
        self.mdl_checkpoint_dir = checkpoint_dir
        self.mdl_checkpoin_filename = os.path.join(self.mdl_checkpoint_dir, file_name)
        

        #Main Network setup
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calc_conv_out_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions) # the output here is the Q values of each action, given an input state (game frame)

        ## Optimizer, loss function, device setup
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_out_dims(self, input_dims):
        """ utility to calculate the output dimensions of conv layers """
        with torch.no_grad():
            state = torch.zeros(1, *input_dims)
            dims = self.conv1(state)
            dims = self.conv2(dims)
            dims = self.conv3(dims)
            return int(np.prod(dims.size()))
    

    def forward(self, state):
        """ forward pass """

        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2)) # conv3 output is shape batch_size x filters x H x W
        flat = conv3.view(conv3.size()[0], -1)  # flatten 
        fc1 = F.relu(self.fc1(flat))
        actions = self.fc2(fc1)

        return actions
    
    def save_checkpoint(self):
        print("...saving checkpoint...")
        os.makedirs(os.path.dirname(self.mdl_checkpoin_filename), exist_ok=True)
        torch.save(self.state_dict(), self.mdl_checkpoin_filename)

    def load_checkpoint(self):
        print("...loading checkpoint...")
        self.load_state_dict(torch.load(self.mdl_checkpoin_filename))
    
