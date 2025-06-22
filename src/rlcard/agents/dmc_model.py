import numpy as np

import torch
from torch import nn


class CardwiseDMCNet(nn.Module):
    def __init__(self, state_shape, action_shape, mlp_layers=[512, 512, 512]):
        super().__init__()
        self.card_encoder = nn.Sequential(
            nn.Linear(state_shape[1], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.mlp_input_dim = 40 * 16 + np.prod(action_shape)
        layer_dims = [self.mlp_input_dim] + mlp_layers

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_dims[-1], 1))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, obs, actions):
        B = obs.size(0)
        cards_encoded = self.card_encoder(obs)
        obs_flat = cards_encoded.view(B, -1)
        x = torch.cat((obs_flat, actions), dim=1)
        return self.fc_layers(x).flatten()


class DMCNet(nn.Module):
    def __init__(
            self,
            state_shape,
            action_shape,
            mlp_layers=[512, 512, 512, 512, 512]
    ):
        super().__init__()
        input_dim = np.prod(state_shape) + np.prod(action_shape)
        layer_dims = [input_dim] + mlp_layers
        fc = []
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], 1))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, obs, actions):
        obs = torch.flatten(obs, 1)
        actions = torch.flatten(actions, 1)
        x = torch.cat((obs, actions), dim=1)
        values = self.fc_layers(x).flatten()
        return values


class DMCAgent:
    def __init__(
            self,
            state_shape,
            action_shape,
            mlp_layers=[512, 512, 512, 512, 512],
            exp_epsilon=0.01,
            device="0",
    ):
        self.use_raw = False
        self.device = 'cuda:' + device if device != "cpu" else "cpu"
        self.net = CardwiseDMCNet(state_shape, action_shape, mlp_layers).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape

    def step(self, state):
        action_keys, values = self.predict(state)

        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon:
            action = np.random.choice(action_keys)
        else:
            action_idx = np.argmax(values)
            action = action_keys[action_idx]

        return action

    def eval_step(self, state):
        action_keys, values = self.predict(state)

        action_idx = np.argmax(values)
        action = action_keys[action_idx]

        info = {}
        info['values'] = {state['legal_actions'][i]: float(values[i]) for i in range(len(action_keys))}

        return action, info

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def predict(self, state):
        # Prepare obs and actions
        obs = state['obs'].astype(np.float32)
        legal_actions = state['legal_actions']
        action_values = list()
        # One-hot encoding if there is no action features
        for i in range(len(action_values)):
            action_values.append(np.zeros(self.action_shape[0]))
            action_values[i][legal_actions[i]] = 1
        action_values = np.array(action_values, dtype=np.float32)

        obs = np.repeat(obs[np.newaxis, :], len(legal_actions), axis=0)

        # Predict Q values
        values = self.net.forward(torch.from_numpy(obs).to(self.device),
                                  torch.from_numpy(action_values).to(self.device))

        return legal_actions, values.cpu().detach().numpy()

    def forward(self, obs, actions):
        return self.net.forward(obs, actions)

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = device


class DMCModel:
    def __init__(
            self,
            state_shape,
            action_shape,
            mlp_layers=[512, 512, 512, 512, 512],
            exp_epsilon=0.01,
            device=0
    ):
        self.agents = []
        for player_id in range(len(state_shape)):
            agent = DMCAgent(
                state_shape[player_id],
                action_shape[player_id],
                mlp_layers,
                exp_epsilon,
                device,
            )
            self.agents.append(agent)

    def share_memory(self):
        for agent in self.agents:
            agent.share_memory()

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def parameters(self, index):
        return self.agents[index].parameters()

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents
