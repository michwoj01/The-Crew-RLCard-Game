import sys

from src.rlcard.agents.human_gui_agent import HumanAgentGUI
from src.rlcard.agents.mcts_agent import MCTSAgent
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed


def run_game(no_tasks):
    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env = make('crew', config={
        'num_players': 4,
        'no_tasks': no_tasks,
        'skip_signals': False,
        'algorithm': 'mcts',
        'is_clone': False,
    })

    set_seed(42)

    human_agent = HumanAgentGUI()
    agents = [human_agent]
    for i in range(env.num_players - 1):
        agents.append(MCTSAgent(env, n_simulations=1000))

    env.set_agents(agents)

    trajectories, payoffs = env.run(is_training=False)

    if all(p > 0 for p in payoffs):
        print("🎉 Team won! All tasks completed successfully!")
    elif all(p < 0 for p in payoffs):
        print("😞 Team lost. Tasks were not completed correctly.")
    else:
        print("🤔 Mixed results - some tasks completed.")


if __name__ == '__main__':
    run_game(int(sys.argv[1]))
