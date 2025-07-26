# From https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Uses hyperparamters and more code from there

import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

# Setup

# Set up a replay memory & a transition class for DNN training:
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# the actual DQN model:
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net.forward(x)

    def save_state(self, name="dqn_model.pth"):
        """Save the model state to a file."""
        torch.save(self.state_dict(), name)

    def load_state(self, name="dqn_model.pth"):
        """Load the model state from a file."""
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(name, map_location=device))


# Hyperparameters
BATCH_SIZE = 512
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
DECAY_FACTOR = -1.0 / EPS_DECAY
TAU = 0.05


class DQNAgent:
    def __init__(self, n_observations, n_actions, device, load_model=False):
        self.device = device
        # Initialize the DQN model, target model, optimizer, and replay memory
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        # check if we can load a saved model
        self.steps_done = 0

        if load_model:
            try:
                self.policy_net.load_state("dqn_model.pth")
                print("Loaded saved model.")
                self.target_net.load_state("dqn_target_model.pth")
                print("Loaded saved target model.")
                self.steps_done = math.inf
            except FileNotFoundError:
                print("No saved model found, starting from scratch.")
                self.target_net.load_state_dict(self.policy_net.state_dict())  # poor man's deep copy

        self.optimizer = optim.AdamW(self.policy_net.parameters(), amsgrad=True, fused=True)
        self.memory = ReplayMemory(10_000)

    def select_action(self, env, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(self.steps_done * DECAY_FACTOR)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)

    def greedy_select_action(self, state):
        return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Double DQN next‐state values
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # 1) select best actions by policy_net
            next_q_policy = self.policy_net(non_final_next_states)  # [N, A]
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)  # [N, 1]

            # 2) evaluate them by target_net
            next_q_target = self.target_net(non_final_next_states)  # [N, A]
            target_values = next_q_target.gather(1, next_actions).squeeze(1)  # [N]

            # put them back into full batch tensor
            next_state_values[non_final_mask] = target_values

        # TD target
        expected_state_action_values = reward_batch + (GAMMA * next_state_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # blugh
        self.optimizer.step()

    def model_update(self, state, action, next_state, reward):
        # Store the transition in memory
        self.memory.push(state, action, next_state, torch.tensor([reward], device=self.device))

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights (Polyak Averaging)
        # θ′ ← τ θ + (1 −τ )θ′
        policy_net_state_dict = self.policy_net.state_dict()
        if self.steps_done % 1_000 == 0:
            self.target_net.load_state_dict(policy_net_state_dict)
        else:
            target_net_state_dict = self.target_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            self.target_net.load_state_dict(target_net_state_dict)

    def save_state(self):
        """Save the model state to a file."""
        self.policy_net.save_state("dqn_model.pth")
        self.target_net.save_state("dqn_target_model.pth")
