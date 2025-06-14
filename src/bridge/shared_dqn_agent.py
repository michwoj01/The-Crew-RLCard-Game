import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from rlcard.utils.utils import remove_illegal
from rlcard.utils.utils import get_device


class SharedDQNModel(nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()
        input_dim = np.prod(state_shape)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SharedDQNAgent():
    def __init__(
        self,
        state_shape,
        num_actions,
        lr=1e-3,
        batch_size=64,
        replay_memory_size=10000,
        min_replay_size=500,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=10000,
        device=None,
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device or get_device()
        self.use_raw = False

        self.model = SharedDQNModel(state_shape, num_actions).to(self.device)
        self.target_model = SharedDQNModel(state_shape, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.replay_memory = []
        self.replay_memory_size = replay_memory_size
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

    def step(self, state):
        legal_actions = state['legal_actions']
        state_tensor = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)

        self.step_count += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

        if np.random.rand() < self.epsilon:
            return np.random.choice(list(legal_actions.keys()))

        with torch.no_grad():
            q_values = self.model(state_tensor)[0].cpu().numpy()

        q_values = remove_illegal(q_values, list(legal_actions.keys()))
        return np.argmax(q_values)

    def feed(self, ts):
        self.replay_memory.append(ts)
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)

        if len(self.replay_memory) >= self.min_replay_size:
            self.train()

    def train(self):
        batch = random.sample(self.replay_memory, self.batch_size)

        states = [b[0]['obs'] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3]['obs'] if b[3] is not None else np.zeros_like(b[0]) for b in batch]
        dones = [b[4] for b in batch]

        state_batch = torch.tensor(states, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.model(state_batch).gather(1, action_batch)
        with torch.no_grad():
            max_next_q_values = self.target_model(next_state_batch).max(1, keepdim=True)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step_count % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def eval_step(self, state):
        legal_actions = state['legal_actions']
        state_tensor = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor)[0].cpu().numpy()

        q_values = remove_illegal(q_values, list(legal_actions.keys()))
        return np.argmax(q_values)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(self.model.state_dict())
