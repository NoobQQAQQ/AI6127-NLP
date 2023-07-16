import argparse


def get_parameters():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--N', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--no_scale', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.2)

    # Training setting
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--eval_bs', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=35)
    parser.add_argument('--lr', type=float, default=20)

    # Path
    parser.add_argument('--data_path', type=str, default='../data')

    return parser.parse_args()
