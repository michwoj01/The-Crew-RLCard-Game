import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rlcard.agents.dqn_agent import DQNAgent
from src.rlcard.utils import remove_illegal

Transition = collections.namedtuple('Transition', 'info_state action_probs')


class NFSPAgent(object):
    def __init__(self,
                 num_actions=4,
                 state_shape=None,
                 hidden_layers_sizes=None,
                 reservoir_buffer_capacity=20000,
                 anticipatory_param=0.1,
                 batch_size=256,
                 train_every=1,
                 rl_learning_rate=0.1,
                 sl_learning_rate=0.005,
                 min_buffer_size_to_learn=100,
                 q_replay_memory_size=20000,
                 q_replay_memory_init_size=100,
                 q_update_target_estimator_every=1000,
                 q_discount_factor=0.99,
                 q_epsilon_start=0.06,
                 q_epsilon_end=0,
                 q_epsilon_decay_steps=int(1e6),
                 q_batch_size=32,
                 q_train_every=1,
                 q_mlp_layers=None,
                 evaluate_with='average_policy',
                 device=None,
                 save_path=None,
                 save_every=float('inf')):
        self.use_raw = False
        self._num_actions = num_actions
        self._state_shape = state_shape
        self._layer_sizes = hidden_layers_sizes + [num_actions]
        self._batch_size = batch_size
        self._train_every = train_every
        self._sl_learning_rate = sl_learning_rate
        self._anticipatory_param = anticipatory_param
        self._min_buffer_size_to_learn = min_buffer_size_to_learn

        self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None
        self.evaluate_with = evaluate_with

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # Build the action-value network
        self._rl_agent = DQNAgent(q_replay_memory_size, q_replay_memory_init_size,
                                  q_update_target_estimator_every, q_discount_factor, q_epsilon_start, q_epsilon_end,
                                  q_epsilon_decay_steps, q_batch_size, num_actions, state_shape, q_train_every,
                                  q_mlp_layers,
                                  rl_learning_rate, device)

        # Build the average policy supervised model
        self._build_model()

        self.sample_episode_policy()

        # Checkpoint saving parameters
        self.save_path = save_path
        self.save_every = save_every

    def _build_model(self):

        # configure the average policy network
        policy_network = AveragePolicyNetwork(self._num_actions, self._state_shape, self._layer_sizes)
        policy_network = policy_network.to(self.device)
        self.policy_network = policy_network
        self.policy_network.eval()

        # xavier init
        for p in self.policy_network.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # configure optimizer
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self._sl_learning_rate)

    def feed(self, ts):
        self._rl_agent.feed(ts)
        self.total_t += 1
        if self.total_t > 0 and len(
                self._reservoir_buffer) >= self._min_buffer_size_to_learn and self.total_t % self._train_every == 0:
            sl_loss = self.train_sl()
            print('\rINFO - Step {}, sl-loss: {}'.format(self.total_t, sl_loss), end='')

    def step(self, state):
        obs = state['obs']
        legal_actions = state['legal_actions']
        if self._mode == 'best_response':
            action = self._rl_agent.step(state)
            one_hot = np.zeros(self._num_actions)
            one_hot[action] = 1
            self._add_transition(obs, one_hot)

        elif self._mode == 'average_policy':
            probs = self._act(obs)
            probs = remove_illegal(probs, legal_actions)
            action = np.random.choice(len(probs), p=probs)

        return action

    def eval_step(self, state):
        if self.evaluate_with == 'best_response':
            action, info = self._rl_agent.eval_step(state)
        elif self.evaluate_with == 'average_policy':
            obs = state['obs']
            legal_actions = state['legal_actions']
            probs = self._act(obs)
            probs = remove_illegal(probs, legal_actions)
            action = np.random.choice(len(probs), p=probs)
            info = {
                'probs': {legal_actions[i]: float(probs[legal_actions[i]]) for i in
                          range(len(legal_actions))}}
        else:
            raise ValueError("'evaluate_with' should be either 'average_policy' or 'best_response'.")
        return action, info

    def sample_episode_policy(self):
        if np.random.rand() < self._anticipatory_param:
            self._mode = 'best_response'
        else:
            self._mode = 'average_policy'

    def _act(self, info_state):
        info_state = np.expand_dims(info_state, axis=0)
        info_state = torch.from_numpy(info_state).float().to(self.device)

        with torch.no_grad():
            log_action_probs = self.policy_network(info_state).cpu().numpy()

        action_probs = np.exp(log_action_probs)[0]

        return action_probs

    def _add_transition(self, state, probs):
        transition = Transition(
            info_state=state,
            action_probs=probs)
        self._reservoir_buffer.add(transition)

    def train_sl(self):
        if (len(self._reservoir_buffer) < self._batch_size or
                len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
            return None

        transitions = self._reservoir_buffer.sample(self._batch_size)
        info_states = [t.info_state for t in transitions]
        action_probs = [t.action_probs for t in transitions]

        self.policy_network_optimizer.zero_grad()
        self.policy_network.train()

        # (batch, state_size)
        info_states = torch.from_numpy(np.array(info_states)).float().to(self.device)

        # (batch, num_actions)
        eval_action_probs = torch.from_numpy(np.array(action_probs)).float().to(self.device)

        # (batch, num_actions)
        log_forecast_action_probs = self.policy_network(info_states)

        ce_loss = - (eval_action_probs * log_forecast_action_probs).sum(dim=-1).mean()
        ce_loss.backward()

        self.policy_network_optimizer.step()
        ce_loss = ce_loss.item()
        self.policy_network.eval()

        self.train_t += 1

        if self.save_path and self.train_t % self.save_every == 0:
            # To preserve every checkpoint separately, 
            # add another argument to the function call parameterized by self.train_t
            self.save_checkpoint(self.save_path)
            print("\nINFO - Saved model checkpoint.")

        return ce_loss

    def set_device(self, device):
        self.device = device
        self._rl_agent.set_device(device)

    def checkpoint_attributes(self):
        return {
            'agent_type': 'NFSPAgent',
            'policy_network': self.policy_network.checkpoint_attributes(),
            'reservoir_buffer': self._reservoir_buffer.checkpoint_attributes(),
            'rl_agent': self._rl_agent.checkpoint_attributes(),
            'policy_network_optimizer': self.policy_network_optimizer.state_dict(),
            'device': self.device,
            'anticipatory_param': self._anticipatory_param,
            'batch_size': self._batch_size,
            'min_buffer_size_to_learn': self._min_buffer_size_to_learn,
            'num_actions': self._num_actions,
            'mode': self._mode,
            'evaluate_with': self.evaluate_with,
            'total_t': self.total_t,
            'train_t': self.train_t,
            'sl_learning_rate': self._sl_learning_rate,
            'train_every': self._train_every,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        print("\nINFO - Restoring model from checkpoint...")
        agent = cls(
            anticipatory_param=checkpoint['anticipatory_param'],
            batch_size=checkpoint['batch_size'],
            min_buffer_size_to_learn=checkpoint['min_buffer_size_to_learn'],
            num_actions=checkpoint['num_actions'],
            sl_learning_rate=checkpoint['sl_learning_rate'],
            train_every=checkpoint['train_every'],
            evaluate_with=checkpoint['evaluate_with'],
            device=checkpoint['device'],
            q_mlp_layers=checkpoint['rl_agent']['q_estimator']['mlp_layers'],
            state_shape=checkpoint['rl_agent']['q_estimator']['state_shape'],
            hidden_layers_sizes=[],
        )

        agent.policy_network = AveragePolicyNetwork.from_checkpoint(checkpoint['policy_network'])
        agent._reservoir_buffer = ReservoirBuffer.from_checkpoint(checkpoint['reservoir_buffer'])
        agent._mode = checkpoint['mode']
        agent.total_t = checkpoint['total_t']
        agent.train_t = checkpoint['train_t']
        agent.policy_network.to(agent.device)
        agent.policy_network.eval()
        agent.policy_network_optimizer = torch.optim.Adam(agent.policy_network.parameters(), lr=agent._sl_learning_rate)
        agent.policy_network_optimizer.load_state_dict(checkpoint['policy_network_optimizer'])
        agent._rl_agent.from_checkpoint(checkpoint['rl_agent'])
        agent._rl_agent.set_device(agent.device)
        return agent

    def save_checkpoint(self, path, filename='checkpoint_nfsp.pt'):
        torch.save(self.checkpoint_attributes(), path + '/' + filename)


class AveragePolicyNetwork(nn.Module):

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        super(AveragePolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # set up mlp w/ relu activations
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        mlp = [nn.Flatten(), nn.BatchNorm1d(layer_dims[0])]
        for i in range(len(layer_dims) - 1):
            mlp.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:  # all but final have relu
                mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, s):
        logits = self.mlp(s)
        log_action_probs = F.log_softmax(logits, dim=-1)
        return log_action_probs

    def checkpoint_attributes(self):
        return {
            'num_actions': self.num_actions,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'mlp': self.mlp.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        agent = cls(
            num_actions=checkpoint['num_actions'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
        )
        agent.mlp.load_state_dict(checkpoint['mlp'])
        return agent


class ReservoirBuffer(object):
    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def checkpoint_attributes(self):
        return {
            'data': self._data,
            'add_calls': self._add_calls,
            'reservoir_buffer_capacity': self._reservoir_buffer_capacity,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        reservoir_buffer = cls(checkpoint['reservoir_buffer_capacity'])
        reservoir_buffer._data = checkpoint['data']
        reservoir_buffer._add_calls = checkpoint['add_calls']
        return reservoir_buffer

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
