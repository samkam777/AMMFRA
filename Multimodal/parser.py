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
                        default=20,
                        help="training epoches")

    parser.add_argument("--algorithm",
                        type=str,
                        default="BST",
                        choices=["NeuMF", "BST", "Item2Vec", "AutoInt", "DCNV2", "withoutMulti"])

    # get time
    parser.add_argument("--_running_time",
                        type=str,
                        default="2021-00-00-00-00-00",
                        help="running time")


    # set device and parameters
    args = parser.parse_args()

    return args







