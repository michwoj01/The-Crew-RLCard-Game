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
