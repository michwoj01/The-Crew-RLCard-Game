from rlcard.envs.registration import register
import rlcard

from players import NFSPCrewPlayer

register(
    env_id='crew',
    entry_point='env:CrewEnv',
)

env = rlcard.make('crew')

env.set_agents([NFSPCrewPlayer(0),
                NFSPCrewPlayer(1),
                NFSPCrewPlayer(2),
                NFSPCrewPlayer(3)])

for episode in range(1000):
    trajectories, _ = env.run(is_training=True)
    for i, agent in enumerate(env.agents):
        for ts in trajectories[i]:
            if isinstance(ts, (list, tuple)) and len(ts) >= 5:
                state, action, reward, next_state, done = ts[:5]
                agent.feed((state, action, reward, next_state, done))
            else:
                print("Unexpected transition format:", ts)
            agent.feed((state, action, reward, next_state, done))
    if episode % 100 == 0:
        print(f"Episode {episode} done")

trajectories, payoffs = env.run(is_training=False)
print('Final evaluation payoffs:', payoffs)
