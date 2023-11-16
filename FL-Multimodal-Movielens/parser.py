import argparse
from util import str2bool

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Seed")
    parser.add_argument("--lr",
                        type=float,
                        default=0.0005,
                        help="learning rate")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="dropout rate")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="batch size for training")
    parser.add_argument("--epochs",
                        type=int,
                        default=80,
                        help="training epoches")

    parser.add_argument("--lamda",
                        type=float,
                        default=0.0001,
                        help="Regularization term")
    parser.add_argument("--beta_perfed",
                        type=float,
                        default=0.0001,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")

    parser.add_argument("--n_users",
                        type=int,
                        default=50,
                        help="Number of segmentation of data")
    parser.add_argument("--algorithm",
                        type=str,
                        default="FedAvg",
                        choices=["pFedMe", "FedAvg", "perFed", "FedApple"])
    # personal setting
    parser.add_argument("--personal_learning_rate",
                        type=float,
                        default=0.0005,
                        help="Persionalized learning rate to caculate theta aproximately using K steps")

    # DP
    parser.add_argument('--delta',
                        type=float,
                        default=1e-4,
                        help='DP DELTA')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help='DP MAX_GRAD_NORM')
    parser.add_argument('--noise_multiplier',
                        type=float,
                        default=0.1,
                        help='DP NOISE_MULTIPLIER')

    parser.add_argument("--if_DP",
                        type=str2bool,
                        default=True,
                        help="if DP")
    parser.add_argument('--num_select_users_rate',
                        type=float,
                        default=0.5,
                        help='rate of selected users')
    # if balance data
    parser.add_argument("--_balance",
                        type=str2bool,
                        default=False,
                        help="if balance data")

    parser.add_argument('--UBI',
                        type=float,
                        default=100.0,
                        help='UBI')

    # get time
    parser.add_argument("--_running_time",
                        type=str,
                        default="2021-00-00-00-00-00",
                        help="running time")

    parser.add_argument('--beta',
                        type=float,
                        default=1e-8,
                        help='beta')

    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.001)
    parser.add_argument('-L', "--L", type=float, default=20.0)
    parser.add_argument('-mu', "--mu", type=float, default=0.1,
                        help="Proximal rate for FedProx")

    parser.add_argument("--model",
                        type=str,
                        default="BST_Multimodal",
                        choices=["BST_Multimodal", "BST_without", "NeuMF", "Item2Vec", "AutoInt", "DCNV2"])



    # set device and parameters
    args = parser.parse_args()

    return args