import torch
import util
import time
from parser import parser
from dataset_split import get_data
from model import BST
from FLAlgorithms.server.serveravg import FedAvg
from torch.utils.tensorboard import SummaryWriter

from AutoInt import AutoInt
from DCN_V2 import DCNV2
from NeuMF import NeuMF
from item2vec import Item2Vec
from without_multi import BST_without

import os
import warnings
warnings.filterwarnings("ignore")

def main():
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # seed for Reproducibility
    util.seed_everything(args.seed)

    # hyper-parameter
    hyper_param = "_user_" + str(args.n_users) + "_algorithm_" + str(args.algorithm) + "_DP_" + str(args.if_DP) + "_model_" + str(args.model) +  "_"
    print("hyper_param: {}\t".format(hyper_param))

    # running time
    if args._running_time != "2021-00-00-00-00-00":
        running_time = args._running_time  # get time from *.sh file
    else:
        running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # get the present time
    print("running time: {}".format(running_time))


    # model
    if args.model == "BST_Multimodal":
        # dataset
        users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data(args)
        model = BST(users, ratings, movies, genres, device=device)
        criterion = torch.nn.BCELoss()
    elif args.model == "NeuMF":
        users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data(args)
        model = NeuMF(users, ratings, movies, genres, device=device)
        criterion = torch.nn.BCELoss()
    elif args.model == "Item2Vec":
        model = Item2Vec()
        local_train_datasets, local_val_datasets, local_test_datasets = model.get_data()
        criterion = model.criterion
    elif args.model == "AutoInt":
        users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data(args)
        model = AutoInt(users, ratings, movies, genres, 256, device)
        criterion = torch.nn.BCELoss()
    elif args.model == "DCNV2":
        users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data(args)
        model = DCNV2(users, movies, device)
        criterion = torch.nn.BCELoss()
    elif args.model == "BST_without":
        users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data(args)
        model = BST_without(users, ratings, movies, genres, device=device)
        criterion = torch.nn.BCELoss()

    model = model.to(device)
    # print(model)
    # print(next(model.parameters()).device)
    # loss function

    path = "/huanggx/0002code/multi_FL_all_V1/"
    tensorboard_path = path+'/' + running_time + "/" + hyper_param + "/" + "tensorboard/"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    if(args.algorithm == "FedAvg"):
        server = FedAvg(device, args, local_train_datasets, local_val_datasets, local_test_datasets, model, running_time, hyper_param, writer, criterion)


    server.train()







if __name__ == '__main__':
    main()

