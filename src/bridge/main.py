from rlcard.envs.registration import register
import rlcard
import pprint
from rlcard.utils import set_seed
from rlcard.agents import RandomAgent

register(
    env_id='crew',
    entry_point='env:CrewEnv',
)

env = rlcard.make('crew', config={'seed': 42})

set_seed(42)

# Set agents
agent = RandomAgent(num_actions=env.num_actions)
env.set_agents([agent for _ in range(env.num_players)])

for epoch in range(1000):
    trajectories, player_wins = env.run(is_training=True)
trajectories, player_wins = env.run(is_training=False)
# Print out the trajectories
print('\nTrajectories:')
print(trajectories)
print('\nSample raw observation:')
pprint.pprint(trajectories[0][0]['raw_obs'])
print('\nSample raw legal_actions:')
pprint.pprint(trajectories[0][0]['raw_legal_actions'])
