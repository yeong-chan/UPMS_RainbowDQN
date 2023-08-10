import argparse

def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--mode", type=str, default='WCOVERT', help="action mode")
    parser.add_argument("--n_job", type=int, default=1000, help="number of job")

    parser.add_argument("--n_episode", type=int, default=1000000, help="number of episodes")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping paramter")
    parser.add_argument("--K_epoch", type=int, default=1, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=50, help="running horizon")
    parser.add_argument("--w_delay", type=float, default=1.0, help="weight for minimizing delays")
    parser.add_argument("--w_move", type=float, default=0.5, help="weight for minimizing the number of ship movements")
    parser.add_argument("--w_priority", type=float, default=0.5, help="weight for maximizing the efficiency")
    return parser.parse_args()