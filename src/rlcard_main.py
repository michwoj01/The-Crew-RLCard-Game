from rlcard.envs.registration import register
import rlcard

register(
    env_id='crew',
    entry_point='rlcard_env:CrewRLCardEnv',
)

env = rlcard.make('crew')
