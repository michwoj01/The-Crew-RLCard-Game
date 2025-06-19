from rlcard.envs.registration import register
import rlcard
from rlcard.utils import set_seed
from rlcard.agents import RandomAgent

from src.bridge.human_player import HumanAgent

register(
    env_id='crew',
    entry_point='env:CrewEnv',
)

env = rlcard.make('crew', config={
    'num_players': 4,
    'no_tasks': 2,
    'seed': 42,
    'fixed_tasks': False,
    'skip_signals': False,
})

set_seed(42)

# Set agents
agent = RandomAgent(num_actions=env.num_actions)
agents = [agent for _ in range(env.num_players - 1)]
agents.insert(0, HumanAgent(197))
env.set_agents(agents)

trajectories, player_wins = env.run(is_training=False)

print(player_wins)
