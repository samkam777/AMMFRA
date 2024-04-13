

#

'''
'''




# attention分开四个


import util
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
# import torchmetrics
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import random
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from parser import parser
from NeuMF import NeuMF
from item2vec import Item2Vec
from logging_result import result_logging
import torch.nn.functional as F
from Autont import AutoInt
from DCN_V2 import DCNV2

import warnings
warnings.filterwarnings("ignore")

data_path = "./data/"



def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=True



def image_feature(img_feature_df, movie_history, sequence):

    total_img_name = movie_history[0].item() if sequence else movie_history
    img_f = eval(
        str(img_feature_df.loc[img_feature_df['image_name'] == total_img_name, 'image_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
    total_out = torch.tensor(img_f, dtype=torch.float32)
    # print(total_out.shape)

    if sequence:
        total_out = torch.unsqueeze(total_out, dim=0)
        for idx, item in enumerate(movie_history[1:]):
            img = eval(str(img_feature_df.loc[img_feature_df['image_name'] == item.item(), 'image_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
            img = torch.tensor(img, dtype=torch.float32)
            out = torch.unsqueeze(img, dim=0)
            total_out = torch.cat((total_out, out), dim=0)

    return total_out

def txt_feature(txt_feature_tensor, movie_history, sequence):
    # total_txt_name = movie_history[0].item() if sequence else movie_history
    # total_out = txt_feature_tensor[total_txt_name]
    # print(total_out.shape)
    #
    # if sequence:
    #     total_out = torch.unsqueeze(total_out, dim=0)
    #     for idx, item in enumerate(movie_history[1:]):
    #         txt = txt_feature_tensor[item]
    #         out = torch.unsqueeze(txt, dim=0)
    #         total_out = torch.cat((total_out, out), dim=0)
    #
    # return total_out
    total_txt_name = movie_history[0].item() if sequence else movie_history
    txt_f = eval(
        str(txt_feature_tensor.loc[txt_feature_tensor['txt_name'] == total_txt_name, 'txt_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
    total_out = torch.tensor(txt_f, dtype=torch.float32)
    # print(total_out.shape)

    if sequence:
        total_out = torch.unsqueeze(total_out, dim=0)
        for idx, item in enumerate(movie_history[1:]):
            txt = eval(str(
                txt_feature_tensor.loc[txt_feature_tensor['txt_name'] == item.item(), 'txt_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
            txt = torch.tensor(txt, dtype=torch.float32)
            out = torch.unsqueeze(txt, dim=0)
            total_out = torch.cat((total_out, out), dim=0)

    return total_out


def genres_feature(genres_feature_df, movie_history, sequence):
    total_movie_name = movie_history[0].item() if sequence else movie_history
    # print(total_movie_name)
    movie_genres_f = eval(str(genres_feature_df.loc[genres_feature_df["itemid"] == total_movie_name, "encoded_cat"].str.replace("[", "", regex=True).str.replace("]","", regex=True).str.replace("\n", "", regex=True).str.replace(".", ",", regex=True).item()))
    total_out = torch.tensor(movie_genres_f, dtype=torch.float32)

    if sequence:
        total_out = torch.unsqueeze(total_out, dim=0)
        for idx, item in enumerate(movie_history[1:]):
            genres = eval(str(genres_feature_df.loc[genres_feature_df["itemid"] == item.item(), "encoded_cat"].str.replace("[", "", regex=True).str.replace("]","", regex=True).str.replace("\n", "", regex=True).str.replace(".", ",", regex=True).item()))
            genres = torch.tensor(genres, dtype=torch.float32)
            out = torch.unsqueeze(genres, dim=0)
            total_out = torch.cat((total_out, out), dim=0)

    return total_out

# class MovieDataset(Dataset):
#     def __init__(self, ratings_file, test=False):
#         self.ratings_frame = pd.read_csv(ratings_file, delimiter=",")
#         self.test = test
#         # self.img_feature_df = pd.read_csv("./data/img_feature/img_feature_1024_V2.csv")
#
#     def __len__(self):
#         return len(self.ratings_frame)
#
#     def __getitem__(self, idx):
#         data = self.ratings_frame.iloc[idx]
#         user_id = data.user_id
#
#         movie_history = eval(data.sequence_movie_ids)  # if sequence_movie_ids = 3186,1721,1270,1022,2340,1836,3408,1207; movie_history = (3186, 1721, 1270, 1022, 2340, 1836, 3408, 1207)
#         # print(movie_history)
#         movie_history_ratings = eval(data.sequence_ratings)
#         # print(movie_history_ratings)
#         movie_id = movie_history[-1:][0]  # movie_history is tuple, [0] takes the first element
#         rating = movie_history_ratings[-1:][0]
#         # print("movie_id :{}\t user_id:{}".format(movie_id, user_id))
#
#
#         sex = data.sex
#         age_group = data.age_group
#         occupation = data.occupation
#
#         # return user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_image_feature, target_movie_id_image_feature
#         # return user_id, movie_id, rating, movie_image_feature, sex, age_group, occupation
#         return user_id, movie_id, rating, sex, age_group, occupation


class MovieDataset(Dataset):
    def __init__(self, ratings_file, test=False):
        self.ratings_frame = pd.read_csv(ratings_file, delimiter=",")
        self.test = test
        self.img_feature_df = pd.read_csv(data_path + "/img_feature/img_feature_1024_V2.csv")
        self.genres_feature_df = pd.read_csv(data_path + "/movies_genres.csv", sep=",")
        # print(self.genres_feature_df)
        # self.txt_feature_tensor = torch.load("/home/jadeting/samkam/code/dataset/data/bert_feature/bert_extract.pth")
        self.txt_feature_df = pd.read_csv(data_path + "/bert_feature/bert_extract.csv")

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        movie_history = eval(
            data.sequence_movie_ids)  # if sequence_movie_ids = 3186,1721,1270,1022,2340,1836,3408,1207; movie_history = (3186, 1721, 1270, 1022, 2340, 1836, 3408, 1207)
        movie_history_ratings = eval(data.sequence_ratings)
        target_movie_id = movie_history[-1:][0]  # movie_history is tuple, [0] takes the first element
        target_movie_rating = movie_history_ratings[-1:][0]

        movie_history = torch.LongTensor(movie_history[:-1])  # [:-1] except for the last one
        movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])


        # movie_history_image_feature = image_feature(self.img_feature_df, movie_history, True)
        target_movie_id_image_feature = image_feature(self.img_feature_df, target_movie_id, False)
        # print(movie_history_image_feature)

        # target movie txt bert feature
        target_movie_id_txt_feature = txt_feature(self.txt_feature_df, target_movie_id, False)


        movie_history_genres = genres_feature(self.genres_feature_df, movie_history, True)
        target_movie_genres = genres_feature(self.genres_feature_df, target_movie_id, False)
        # print("target_movie_genres : {}".format(target_movie_genres))

        sex = data.sex
        age_group = data.age_group
        occupation = data.occupation

        return user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature




def get_data(batch_size):
    users = pd.read_csv(data_path + "/users.csv", sep=",")
    ratings = pd.read_csv(data_path + "/ratings.csv", sep=",")
    movies = pd.read_csv(data_path + "/movies.csv", sep=",")
    genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    for genre in genres:
        movies[genre] = movies["genres"].apply(lambda values: int(genre in values.split("|")))

    train_dataset = MovieDataset(data_path + "/new_dataset4/train_data.csv")
    # train_dataset = MovieDataset(data_path + "/train_data.csv")
    val_dataset = MovieDataset(data_path + "/new_dataset4/val_data.csv")
    test_dataset = MovieDataset(data_path + "/new_dataset4/test_data.csv")

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=4)
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=4)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=4)

    return users, ratings, movies, genres, train_data, val_data, test_data

class AdditiveAttention(nn.Module):
    '''
    A general additive attention module
    '''
    def __init__(self, candidate_vector_dim, query_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector):
        '''
        :param candidate_vector:    batch_size, candidate_size, candidate_vector_dim
        :return:                    batch_size, candidate_vector_dim
        '''

        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))

        # print("temp shape:{}".format(temp.shape))

        # print("torch.matmul(temp, self.attention_query_vector) shape:{}".format(torch.matmul(
        #     temp, self.attention_query_vector).shape))


        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector), dim=1)     # 128, 1

        # print("candidate_weights shape:{}".format(candidate_weights.shape)) # 128, 1

        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)


        # print("target shape:{}".format(target.shape))

        return target


class BST(nn.Module):
    def __init__(self, users, ratings, movies, genres, device, args=None):
        super(BST, self).__init__()
        self.users = users
        self.ratings = ratings
        self.movies = movies
        self.genres = genres
        self.device = device
        self.args = args

        # embedding layer
        # user
        self.embeddings_user_id = nn.Embedding(int(self.users.user_id.max()) + 1,
                                               int(math.sqrt(self.users.user_id.max())) + 1)            # Embedding(6041, 78)
        self.embeddings_user_sex = nn.Embedding(len(self.users.sex.unique()),
                                                int(math.sqrt(len(self.users.sex.unique()))))           # Embedding(2, 1)
        self.embeddings_age_group = nn.Embedding(len(self.users.age_group.unique()),
                                                 int(math.sqrt(len(self.users.age_group.unique()))))    # Embedding(7, 2)
        self.embeddings_user_occupation = nn.Embedding(len(self.users.occupation.unique()),
                                                       int(math.sqrt(len(self.users.occupation.unique())))) # Embedding(21, 4)

        # movie
        movie_embedding_dims = int(math.sqrt(self.movies.movie_id.max())) + 1           # 63
        self.embeddings_movie_id = nn.Embedding(int(self.movies.movie_id.max()) + 1,
                                                movie_embedding_dims)         # Embedding(3953, 63)
        # Create a vector lookup for movie genres.
        genres_embedding_dims = 73      # ont-hot码73维


        self.transformerlayer = nn.TransformerEncoderLayer(136, 4, dropout=0.2)
        # self.transformerlayer = nn.TransformerEncoderLayer(90, 3, dropout=0.2)

        self.linear = nn.Sequential(#nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(0.2),
                                    # nn.Linear(768, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(0.2),
                                    nn.Linear(768, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.2),
                                    nn.Linear(256, 1), )

        self.img_linear = nn.Sequential(nn.Linear(1024, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256),)

        self.txt_linear = nn.Sequential(nn.Linear(768, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256),)

        self.transformer_linear = nn.Sequential(nn.Linear(544, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256), )

        self.user_linear = nn.Sequential(nn.Linear(141, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(512, 256), )

        self.logistic = nn.Sigmoid()


        self.attention_layer_image = AdditiveAttention(256, 256)
        self.attention_layer_txt = AdditiveAttention(256, 256)
        self.attention_layer_user = AdditiveAttention(256, 256)
        self.attention_layer_transformer = AdditiveAttention(256, 256)


    def encode_input(self, inputs):
        # len: 1, 7, 1, 7, 1, 1, 1, 1

        user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature= inputs


        target_movie_emb = self.embeddings_movie_id(target_movie_id)        # Batchsize dim   # torch.size([128, 63])
        user_id = self.embeddings_user_id(user_id)                          # Batchsize dim   # torch.size([128, 78])
        user_features = torch.cat((user_id, target_movie_emb), 1)           # Batchsize dim   # torch.size([128, 141])

        # behavior
        movie_history_id_emb = self.embeddings_movie_id(movie_history)      # 128, 7, 63
        movie_history_emb = torch.cat((movie_history_id_emb, movie_history_genres), dim=2) # 128, 7, 63+73 = 136

        target_movie_emb = torch.cat((target_movie_emb, target_movie_genres), dim=1)# 128, 63+73=136
        target_movie_emb = target_movie_emb.unsqueeze(1)  # 128, 1, 136
        # target_movie_genres = target_movie_genres.unsqueeze(1)  # 128, 1, 73
        # print("movie_history_genres.shape: {}".format(movie_history_genres.shape))
        # print("target_movie_genres.shape: {}".format(target_movie_genres.shape))
        transformer_features = torch.cat((movie_history_emb, target_movie_emb), dim=1) # 128, 8, 136


        return transformer_features, user_features, target_movie_rating.float(), target_movie_id_image_feature, target_movie_id_txt_feature

    def forward(self, x):
        # transformer_features, user_features, target_movie_rating, image_transformer_features = self.encode_input(x)
        transformer_features, user_features, target_movie_rating, target_movie_id_image_feature, target_movie_id_txt_feature = self.encode_input(x)
        # output = 0

        transformer_output = self.transformerlayer(transformer_features)
        # print("transformer_output : {}".format(transformer_output.shape))               # transformer_output : torch.Size([128, 8, 63])


        transformer_output = torch.flatten(transformer_output, start_dim=1)
        # print("transformer_output : {}".format(transformer_output.shape))               # transformer_output : torch.Size([128, 8*136=1088])
        transformer_output = self.transformer_linear(transformer_output)
        transformer_output = transformer_output.unsqueeze(1)
        transformer_output = self.attention_layer_transformer(transformer_output)

        # Concat with other features
        # features = torch.cat((transformer_output, user_features), dim=1)
        # print("features : {}".format(features.shape))                                   # features : torch.Size([128, 1088+141=1229])

        # img_features = self.img_linear(target_movie_id_image_feature)                     # 128, 256
        # img_features = img_features.unsqueeze(1)                                          # 128, 1, 256
        # img_features = self.attention_layer_image(img_features)                                 # 128, 256

        txt_features = self.txt_linear(target_movie_id_txt_feature)                       # 128, 256
        txt_features = txt_features.unsqueeze(1)
        txt_features = self.attention_layer_txt(txt_features)                                 # 128, 256

        user_features = self.user_linear(user_features)
        user_features = user_features.unsqueeze(1)
        user_features = self.attention_layer_user(user_features)



        # MFB_out, _ = self.MFB(target_movie_id_image_feature, target_movie_id_txt_feature)   # 128, 1000

        features = torch.cat((user_features, transformer_output, txt_features), dim=1)                             # 128, 256*4=1024

        output = self.linear(features)

        output = self.logistic(output)

        return output, target_movie_rating

def compute_metrics(y_true, y_pred):

    auc = roc_auc_score(y_true, y_pred)
    y_pred = (y_pred > 0.5).astype(np.float32)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    return auc, f1, acc, recall, precision


def train(args):

    # seed for Reproducibility
    util.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)



    if args.algorithm == "BST":
        users, ratings, movies, genres, train_data, val_data, test_data = get_data(args.batch_size)
        model = BST(users, ratings, movies, genres, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCELoss()
    elif args.algorithm == "NeuMF":
        users, ratings, movies, genres, train_data, val_data, test_data = get_data(args.batch_size)
        # num_users = ratings['user_id'].nunique()
        # num_items = ratings['movie_id'].nunique()
        model = NeuMF(users, ratings, movies, genres, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCELoss()
    elif args.algorithm == "Item2Vec":
        model = Item2Vec()
        train_data, val_data, test_data = model.get_data()
        optimizer = model.optimizer
        criterion = model.criterion
    elif args.algorithm == "AutoInt":
        users, ratings, movies, genres, train_data, val_data, test_data = get_data(args.batch_size)
        # users, ratings, movies, genres, embedding_size, device, args=None, dropout = 0.5
        model = AutoInt(users, ratings, movies, genres, 256, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCELoss()
    elif args.algorithm == "DCNV2":
        users, ratings, movies, genres, train_data, val_data, test_data = get_data(args.batch_size)
        # users, movies, device, dnn_hidden_units=(128,128,), cross_param='vector', args=None, dropout = 0.5, l2_reg=0.00001
        model = DCNV2(users, movies, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = torch.nn.BCELoss()

    model = model.to(device)

    # hyper-parameter
    hyper_param = "_alg_" + str(args.algorithm) + "_exp_" + str(args.exp) + "_"
    print("hyper_param: {}\t".format(hyper_param))

    # running time
    if args._running_time != "2021-00-00-00-00-00":
        running_time = args._running_time  # get time from *.sh file
    else:
        running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # get the present time
    print("running time: {}".format(running_time))

    tensorboard_path = './' + running_time + "/" + hyper_param + "/" + "tensorboard/"
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    writer = SummaryWriter(tensorboard_path)

    model.train()
    optimizer.zero_grad()

    best_auc = 0


    print("============== start training ==============")
    start_time = time.time()

    for epoch in range(args.epochs):

        train_losses, train_auc, train_f1, train_acc, train_recall, train_precision = [], [], [], [], [], []
        epoch_start_time = time.time()
        for datas in train_data:

            datas = [data.to(device) for data in datas]         # datas is a list, len = 8

            out, target_movie_rating = model(datas)
            # print("out : {}".format(out.shape))                 # torch.Size([128, 1])
            out = out.flatten()

            target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
            target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

            # print("out:{}\ttarget:{}".format(out, target_movie_rating))

            loss = criterion(out, target_movie_rating)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())

            auc, f1, acc, recall, precision = compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                                   out.cpu().detach().numpy())
            train_auc.append(auc)
            train_f1.append(f1)
            train_acc.append(acc)
            train_recall.append(recall)
            train_precision.append(precision)

        epoch_end_time = time.time()

        print(
            "train epoch : {}\tloss : {:.4f}\tauc : {:.4f}\tf1: {:.4f}\tacc :{:.4f}\trecall :{:.4f}\tprecision :{:.4f}\tspend : {:.4f} h".format(epoch,
                sum(train_losses) / len(train_losses), sum(train_auc) / len(train_auc), sum(train_f1) / len(train_f1),
                sum(train_acc) / len(train_acc), sum(train_recall) / len(train_recall),
                sum(train_precision) / len(train_precision), (epoch_start_time - start_time) / 60 / 60))

        model.eval()
        val_auc, val_losses, val_f1, val_acc, val_recall, val_precision = [], [], [], [], [], []

        for datas in val_data:

            datas = [data.to(device) for data in datas]
            with torch.no_grad():
                out, target_movie_rating = model(datas)
                out = out.flatten()

                target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
                target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

                loss = criterion(out, target_movie_rating)
                val_losses.append(loss.item())

                auc, f1, acc, recall, precision = compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                                       out.cpu().detach().numpy())

                val_auc.append(auc)
                val_f1.append(f1)
                val_acc.append(acc)
                val_recall.append(recall)
                val_precision.append(precision)


        print("test \tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(sum(val_losses) / len(val_losses), sum(val_auc) / len(val_auc), sum(val_f1) / len(val_f1), sum(val_acc) / len(val_acc), sum(val_recall) / len(val_recall),
                       sum(val_precision) / len(val_precision)))

        result_logging(epoch, sum(val_losses) / len(val_losses), sum(val_auc) / len(val_auc), sum(val_f1) / len(val_f1), sum(val_acc) / len(val_acc), sum(val_recall) / len(val_recall),
                       sum(val_precision) / len(val_precision), running_time, hyper_param, writer)

        # save best model
        best_model_path = './' + running_time + "/" + hyper_param + "/" + '/logs/best_model/'
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        if (sum(val_auc) / len(val_auc)) > best_auc:
            best_auc = sum(val_auc) / len(val_auc)
            best_model = model.state_dict()
            torch.save(best_model, best_model_path + 'best_model.pth')

    print("Finished Training")

    model.load_state_dict(torch.load(best_model_path + 'best_model.pth'))
    model.eval()

    test_auc, test_losses, test_f1, test_acc, test_recall, test_precision = [], [], [], [], [], []
    for datas in test_data:
        datas = [data.to(device) for data in datas]
        with torch.no_grad():
            out, target_movie_rating = model(datas)
            out = out.flatten()

            target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
            target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

            loss = criterion(out, target_movie_rating)
            test_losses.append(loss.item())

            auc, f1, acc, recall, precision = compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                              out.cpu().detach().numpy())

            test_auc.append(auc)
            test_f1.append(f1)
            test_acc.append(acc)
            test_recall.append(recall)
            test_precision.append(precision)
    print("test \tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(
        sum(test_losses) / len(test_losses), sum(test_auc) / len(test_auc), sum(test_f1) / len(test_f1),
        sum(test_acc) / len(test_acc), sum(test_recall) / len(test_recall),
        sum(test_precision) / len(test_precision)))

    # writer.add_scalar(tag="loss/test", scalar_value=sum(test_losses) / len(test_losses), global_step=epoch)
    # writer.add_scalar(tag="auc/test", scalar_value=sum(test_auc) / len(test_auc), global_step=epoch)
    # writer.add_scalar(tag="f1/test", scalar_value=sum(test_f1) / len(test_f1), global_step=epoch)
    # writer.add_scalar(tag="acc/test", scalar_value=sum(test_acc) / len(test_acc), global_step=epoch)
    # writer.add_scalar(tag="recall/test", scalar_value=sum(test_recall) / len(test_recall), global_step=epoch)
    # writer.add_scalar(tag="precision/test", scalar_value=sum(test_precision) / len(test_precision), global_step=epoch)

    writer.close()



if __name__ == '__main__':
    args = parser()
    train(args)










