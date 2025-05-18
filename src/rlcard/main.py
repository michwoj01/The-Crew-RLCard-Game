from rlcard.envs.registration import register
import rlcard
import pprint

from players import NFSPCrewPlayer

register(
    env_id='crew',
    entry_point='env:CrewEnv',
)

env = rlcard.make('crew', config={'seed': 42})

env.set_agents([NFSPCrewPlayer(0),
                NFSPCrewPlayer(1),
                NFSPCrewPlayer(2),
                NFSPCrewPlayer(3)])


trajectories, player_wins = env.run(is_training=False)
# Print out the trajectories
print('\nTrajectories:')
print(trajectories)
print('\nSample raw observation:')
pprint.pprint(trajectories[0][0]['obs'])
print('\nSample raw legal_actions:')
pprint.pprint(trajectories[0][0]['raw_legal_actions'])
