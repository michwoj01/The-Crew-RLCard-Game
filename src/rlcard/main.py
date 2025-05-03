from rlcard.envs.registration import register
import rlcard

from players import CrewPlayer

register(
    env_id='crew',
    entry_point='env:CrewEnv',
)

env = rlcard.make('crew')

env.set_agents([CrewPlayer(40, 0), CrewPlayer(40, 1), CrewPlayer(40, 2), CrewPlayer(40, 3)])

trajectories, payoffs = env.run(is_training=False)
print('payoffs:', payoffs)