from rlcard.envs.registration import register
import rlcard

from players import CrewRLCardPlayer

register(
    env_id='crew',
    entry_point='env:CrewRLCardEnv',
)

env = rlcard.make('crew')

env.set_agents([CrewRLCardPlayer(40, 0), CrewRLCardPlayer(40, 1), CrewRLCardPlayer(40, 2), CrewRLCardPlayer(40, 3)])

trajectories, payoffs = env.run(is_training=False)
print('payoffs:', payoffs)