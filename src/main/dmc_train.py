import argparse

from src.rlcard.agents import DMCTrainer
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed


def train(args):
    set_seed(args.seed)
    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env = make('crew', config={
        'num_players': args.num_players,
        'no_tasks': args.no_tasks,
        'fixed_tasks': args.fixed_tasks,
        'skip_signals': args.skip_signals,
        'algorithm': args.algorithm,
    })

    trainer = DMCTrainer(
        env=env,
        cuda=args.cuda,
        x_pid=args.x_pid,
        save_dir=args.save_dir,
        total_frames=args.total_frames,
        training_device=args.training_device,
        num_actors=args.num_actors,
        num_buffers=args.num_buffers,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        learning_rate=args.learning_rate,
        save_interval=args.save_interval,
        load_model=args.load_model,
    )
    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_device', type=str, default='0')
    parser.add_argument('--x_pid', type=str, default='crew_dmc')
    parser.add_argument('--save_dir', type=str, default='experiments/dmc_result')
    parser.add_argument('--total_frames', type=int, default=5_000_000)
    parser.add_argument('--num_actors', type=int, default=4)
    parser.add_argument('--num_buffers', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--unroll_length', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=30)
    parser.add_argument('--load_model', action='store_true')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="dmc")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=1)
    parser.add_argument("--fixed_tasks", type=bool, default=True)
    parser.add_argument("--skip_signals", type=bool, default=False)

    args = parser.parse_args()
    train(args)
