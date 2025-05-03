from rlcard.envs.registration import register
from players import CrewPlayer
import rlcard

register(
    env_id='crew',
    entry_point='rlcard_env:CrewRLCardEnv',
)

env = rlcard.make('crew')

env.set_agents([CrewPlayer(0), CrewPlayer(1), CrewPlayer(2), CrewPlayer(3)])

for episode in range(10):
    trajectories, payoffs = env.run(is_training=False)
    print('Episode', episode, 'payoffs:', payoffs)
