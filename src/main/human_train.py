from src.main.human_agent import HumanAgent
from src.rlcard.agents.mcts_agent import MCTSAgent
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed

register(
    env_id='crew',
    entry_point='env:CrewEnv',
)

env = make('crew', config={
    'num_players': 4,
    'no_tasks': 2,
    'skip_signals': False,
    'algorithm': 'mcts',
    'is_clone': False,
})

set_seed(42)

# Set agents
agent = MCTSAgent(env, n_simulations=1000)
agents = [agent for _ in range(env.num_players - 1)]
agents.insert(0, HumanAgent())
env.set_agents(agents)

trajectories, player_wins = env.run(is_training=False)