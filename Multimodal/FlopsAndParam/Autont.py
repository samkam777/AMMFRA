

#

'''
'''







# import util
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
# # import torchmetrics
# import os
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
# import time
# import random
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
# from parser import parser
# from NeuMF import NeuMF
# from item2vec import Item2Vec
# from logging_result import result_logging
import torch.nn.functional as F

# data_path = "/home/jadeting/samkam/code/dataset/data/"
#
#
#
# def seed_torch(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED']=str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.benchmark=True
#         torch.backends.cudnn.deterministic=True
#
#
#
# def image_feature(img_feature_df, movie_history, sequence):
#
#     total_img_name = movie_history[0].item() if sequence else movie_history
#     img_f = eval(
#         str(img_feature_df.loc[img_feature_df['image_name'] == total_img_name, 'image_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
#     total_out = torch.tensor(img_f, dtype=torch.float32)
#     # print(total_out.shape)
#
#     if sequence:
#         total_out = torch.unsqueeze(total_out, dim=0)
#         for idx, item in enumerate(movie_history[1:]):
#             img = eval(str(img_feature_df.loc[img_feature_df['image_name'] == item.item(), 'image_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
#             img = torch.tensor(img, dtype=torch.float32)
#             out = torch.unsqueeze(img, dim=0)
#             total_out = torch.cat((total_out, out), dim=0)
#
#     return total_out
#
# def txt_feature(txt_feature_tensor, movie_history, sequence):
#     # total_txt_name = movie_history[0].item() if sequence else movie_history
#     # total_out = txt_feature_tensor[total_txt_name]
#     # print(total_out.shape)
#     #
#     # if sequence:
#     #     total_out = torch.unsqueeze(total_out, dim=0)
#     #     for idx, item in enumerate(movie_history[1:]):
#     #         txt = txt_feature_tensor[item]
#     #         out = torch.unsqueeze(txt, dim=0)
#     #         total_out = torch.cat((total_out, out), dim=0)
#     #
#     # return total_out
#     total_txt_name = movie_history[0].item() if sequence else movie_history
#     txt_f = eval(
#         str(txt_feature_tensor.loc[txt_feature_tensor['txt_name'] == total_txt_name, 'txt_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
#     total_out = torch.tensor(txt_f, dtype=torch.float32)
#     # print(total_out.shape)
#
#     if sequence:
#         total_out = torch.unsqueeze(total_out, dim=0)
#         for idx, item in enumerate(movie_history[1:]):
#             txt = eval(str(
#                 txt_feature_tensor.loc[txt_feature_tensor['txt_name'] == item.item(), 'txt_feature'].str.replace("[", "", regex=True).str.replace("]", "", regex=True).item()))
#             txt = torch.tensor(txt, dtype=torch.float32)
#             out = torch.unsqueeze(txt, dim=0)
#             total_out = torch.cat((total_out, out), dim=0)
#
#     return total_out
#
#
# def genres_feature(genres_feature_df, movie_history, sequence):
#     total_movie_name = movie_history[0].item() if sequence else movie_history
#     # print(total_movie_name)
#     movie_genres_f = eval(str(genres_feature_df.loc[genres_feature_df["itemid"] == total_movie_name, "encoded_cat"].str.replace("[", "", regex=True).str.replace("]","", regex=True).str.replace("\n", "", regex=True).str.replace(".", ",", regex=True).item()))
#     total_out = torch.tensor(movie_genres_f, dtype=torch.float32)
#
#     if sequence:
#         total_out = torch.unsqueeze(total_out, dim=0)
#         for idx, item in enumerate(movie_history[1:]):
#             genres = eval(str(genres_feature_df.loc[genres_feature_df["itemid"] == item.item(), "encoded_cat"].str.replace("[", "", regex=True).str.replace("]","", regex=True).str.replace("\n", "", regex=True).str.replace(".", ",", regex=True).item()))
#             genres = torch.tensor(genres, dtype=torch.float32)
#             out = torch.unsqueeze(genres, dim=0)
#             total_out = torch.cat((total_out, out), dim=0)
#
#     return total_out
#
#
#
#
# class MovieDataset(Dataset):
#     def __init__(self, ratings_file, test=False):
#         self.ratings_frame = pd.read_csv(ratings_file, delimiter=",")
#         self.test = test
#         self.img_feature_df = pd.read_csv(data_path + "/img_feature/img_feature_1024_V2.csv")
#         self.genres_feature_df = pd.read_csv(data_path + "/movies_genres.csv", sep=",")
#         # print(self.genres_feature_df)
#         # self.txt_feature_tensor = torch.load("/home/jadeting/samkam/code/dataset/data/bert_feature/bert_extract.pth")
#         self.txt_feature_df = pd.read_csv(data_path + "/bert_feature/bert_extract.csv")
#
#     def __len__(self):
#         return len(self.ratings_frame)
#
#     def __getitem__(self, idx):
#         data = self.ratings_frame.iloc[idx]
#         user_id = data.user_id
#
#         movie_history = eval(
#             data.sequence_movie_ids)  # if sequence_movie_ids = 3186,1721,1270,1022,2340,1836,3408,1207; movie_history = (3186, 1721, 1270, 1022, 2340, 1836, 3408, 1207)
#         movie_history_ratings = eval(data.sequence_ratings)
#         target_movie_id = movie_history[-1:][0]  # movie_history is tuple, [0] takes the first element
#         target_movie_rating = movie_history_ratings[-1:][0]
#
#         movie_history = torch.LongTensor(movie_history[:-1])  # [:-1] except for the last one
#         movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])
#
#
#         # movie_history_image_feature = image_feature(self.img_feature_df, movie_history, True)
#         target_movie_id_image_feature = image_feature(self.img_feature_df, target_movie_id, False)
#         # print(movie_history_image_feature)
#
#         # target movie txt bert feature
#         target_movie_id_txt_feature = txt_feature(self.txt_feature_df, target_movie_id, False)
#
#
#         movie_history_genres = genres_feature(self.genres_feature_df, movie_history, True)
#         target_movie_genres = genres_feature(self.genres_feature_df, target_movie_id, False)
#         # print("target_movie_genres : {}".format(target_movie_genres))
#
#         sex = data.sex
#         age_group = data.age_group
#         occupation = data.occupation
#
#         return user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature
#
#
#
#
# def get_data(batch_size):
#     users = pd.read_csv(data_path + "/users.csv", sep=",")
#     ratings = pd.read_csv(data_path + "/ratings.csv", sep=",")
#     movies = pd.read_csv(data_path + "/movies.csv", sep=",")
#     genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
#               "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
#
#     for genre in genres:
#         movies[genre] = movies["genres"].apply(lambda values: int(genre in values.split("|")))
#
#     train_dataset = MovieDataset(data_path + "/new_dataset4/train_data.csv")
#     # train_dataset = MovieDataset(data_path + "/train_data.csv")
#     val_dataset = MovieDataset(data_path + "/new_dataset4/val_data.csv")
#     test_dataset = MovieDataset(data_path + "/new_dataset4/test_data.csv")
#
#     train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                                              num_workers=4)
#     val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                                            num_workers=4)
#     test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                                             num_workers=4)
#
#     return users, ratings, movies, genres, train_data, val_data, test_data
#




class AutoInt(nn.Module):
    def __init__(self, users, ratings, movies, genres, embedding_size, device, args=None, dropout = 0.5):
        super(AutoInt, self).__init__()
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

        embedding_size = int(math.sqrt(self.users.user_id.max())) + 1 + int(math.sqrt(len(self.users.sex.unique()))) + int(
            math.sqrt(len(self.users.age_group.unique()))) + int(
            math.sqrt(len(self.users.occupation.unique()))) + movie_embedding_dims
        

        user_id_param = (int(self.users.user_id.max()) + 1) * (int(math.sqrt(self.users.user_id.max())) + 1)
        user_sex_param = (len(self.users.sex.unique())) * (int(math.sqrt(len(self.users.sex.unique()))))
        age_group_param = (len(self.users.age_group.unique())) * (int(math.sqrt(len(self.users.age_group.unique()))))
        user_occupation_param = (len(self.users.occupation.unique())) * (int(math.sqrt(len(self.users.occupation.unique()))))
        movie_id_param = (int(self.movies.movie_id.max()) + 1) * (movie_embedding_dims)
        total_emd_param = user_id_param+user_sex_param+age_group_param+user_occupation_param+movie_id_param
        # print(f"user id param:{(int(self.users.user_id.max()) + 1) * (int(math.sqrt(self.users.user_id.max())) + 1)}")
        print(f"AutoInt embedding 参数量 : {total_emd_param}")

        self.MultiHead_1 = MultiHead(model_dim=embedding_size, output_dim=embedding_size // 2, num_head=4, dropout=0.5)
        self.MultiHead_2 = MultiHead(model_dim=embedding_size // 2, output_dim=embedding_size // 2, num_head=2, dropout=0.5)
        self.MultiHead_3 = MultiHead(model_dim=embedding_size // 2, output_dim=embedding_size // 2, num_head=2, dropout=0.5)


        self.fc_final = nn.Linear(in_features=embedding_size // 2, out_features=1, bias=False)
        self.dropout = dropout



    def forward(self, user_id, target_movie_id, target_movie_rating, sex, age_group, occupation):

        # user_id, _, target_movie_id, _, target_movie_rating, sex, age_group, occupation, _, _, _, _ = x
        batch_size = user_id.size(0)

        sex = sex.to(torch.long)
        age_group = age_group.to(torch.long)
        occupation = occupation.to(torch.long)
        target_movie_id = target_movie_id.to(torch.long)
        user_id = user_id.to(torch.long)
        sex_emb = self.embeddings_user_sex(sex)
        age_emb = self.embeddings_age_group(age_group)
        occupation_emb = self.embeddings_user_occupation(occupation)

        target_movie_emb = self.embeddings_movie_id(target_movie_id)  # Batchsize dim   # torch.size([128, 63])
        user_id = self.embeddings_user_id(user_id)  # Batchsize dim   # torch.size([128, 78])
        features = torch.cat((user_id, target_movie_emb, sex_emb, age_emb, occupation_emb), 1)  # Batchsize dim   # torch.size([128, 141])

        # print("sex_emb : {}".format(sex_emb.shape))                     # torch.Size([128, 1])
        # print("age_emb : {}".format(age_emb.shape))                     # torch.Size([128, 2])
        # print("occupation_emb : {}".format(occupation_emb.shape))       # torch.Size([128, 4])
        # print("target_movie_emb : {}".format(target_movie_emb.shape))   # torch.Size([128, 63])
        # print("user_id : {}".format(user_id.shape))                     # torch.Size([128, 78])

        data_dim = self.MultiHead_1(features)
        data_dim = F.relu(data_dim)
        # print("data_dim : {}".format(data_dim.shape))
        data_dim = self.MultiHead_2(data_dim)
        data_dim = F.relu(data_dim)
        # print("data_dim : {}".format(data_dim.shape))
        data_dim = self.MultiHead_3(data_dim)
        data_dim = F.relu(data_dim)
        # print("data_dim : {}".format(data_dim.shape))
        data_dim = data_dim.view(batch_size, -1)
        # print("data_dim : {}".format(data_dim.shape))
        # fc and dropout
        data_dim = F.dropout(data_dim)
        output = self.fc_final(data_dim)
        output = torch.sigmoid(output)

        return output, target_movie_rating.float()


class MultiHead(nn.Module):
    def __init__(self, model_dim=256, output_dim=128, num_head=8, dropout=0.5):
        super(MultiHead, self).__init__()
        self.dim_per_head = model_dim // num_head
        self.num_head = num_head
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_head)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_head)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_head)

        # self.product_attention = self.ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(model_dim, output_dim)
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)

    def forward(self, x):
        residual = x
        batch_size = x.size(0)
        key = self.linear_k(x)
        value = self.linear_v(x)
        query = self.linear_q(x)

        # reshape
        key = key.view(batch_size * self.num_head, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_head, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_head, -1, self.dim_per_head)

        # attention
        context = self.ScaledDotProductAttention(query, key, value, 8)
        # concat
        context = context.view(residual.size())
        # residual
        context += residual
        # layer normal
        context = self.layer_norm(context)
        # fc
        context = self.fc(context)

        return context

    def ScaledDotProductAttention(self, q, k, v, scale=None):
        attention = torch.bmm(q, k.transpose(1, 2))  # Q*K
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention)
        context = torch.bmm(attention, v)

        return context

# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, dropout=0.5):
#         self.dropout = dropout
#
#
#     def forward(self, q, k, v, scale=None):
#         attention = torch.bmm(q, k.transpose(1,2))  # Q*K
#         if scale:
#             attention = attention * scale
#         attention = F.softmax(attention, dim=2)
#         attention = F.dropout(attention)
#         context = torch.bmm(attention, v)
#
#         return context


# def compute_metrics(y_true, y_pred):
#
#     auc = roc_auc_score(y_true, y_pred)
#     y_pred = (y_pred > 0.5).astype(np.float32)
#     f1 = f1_score(y_true, y_pred)
#     acc = accuracy_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#
#     return auc, f1, acc, recall, precision


# def train(args):
#
#     # seed for Reproducibility
#     util.seed_everything(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # device = torch.device("cpu")
#     print(device)
#
#
#
#     if args.algorithm == "BST":
#         users, ratings, movies, genres, train_data, val_data, test_data = get_data(args.batch_size)
#         model = BST(users, ratings, movies, genres, device=device)
#         optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
#         criterion = torch.nn.BCELoss()
#     elif args.algorithm == "NeuMF":
#         users, ratings, movies, genres, train_data, val_data, test_data = get_data(args.batch_size)
#         # num_users = ratings['user_id'].nunique()
#         # num_items = ratings['movie_id'].nunique()
#         model = NeuMF(users, ratings, movies, genres, device=device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#         criterion = torch.nn.BCELoss()
#     elif args.algorithm == "Item2Vec":
#         model = Item2Vec()
#         train_data, val_data, test_data = model.get_data()
#         optimizer = model.optimizer
#         criterion = model.criterion
#
#     model = model.to(device)
#
#     # hyper-parameter
#     hyper_param = "_algorithm_" + str(args.algorithm) + "_"
#     print("hyper_param: {}\t".format(hyper_param))
#
#     # running time
#     if args._running_time != "2021-00-00-00-00-00":
#         running_time = args._running_time  # get time from *.sh file
#     else:
#         running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  # get the present time
#     print("running time: {}".format(running_time))
#
#     tensorboard_path = './' + running_time + "/" + hyper_param + "/" + "tensorboard/"
#     if not os.path.exists(tensorboard_path):
#         os.makedirs(tensorboard_path)
#     writer = SummaryWriter(tensorboard_path)
#
#     model.train()
#     optimizer.zero_grad()
#
#     best_auc = 0
#
#
#     print("============== start training ==============")
#     start_time = time.time()
#
#     for epoch in range(args.epochs):
#
#         train_losses, train_auc, train_f1, train_acc, train_recall, train_precision = [], [], [], [], [], []
#         epoch_start_time = time.time()
#         for datas in train_data:
#
#             datas = [data.to(device) for data in datas]         # datas is a list, len = 8
#
#             out, target_movie_rating = model(datas)
#             # print("out : {}".format(out.shape))                 # torch.Size([128, 1])
#             out = out.flatten()
#
#             target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
#             target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1
#
#             loss = criterion(out, target_movie_rating)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             train_losses.append(loss.item())
#
#             auc, f1, acc, recall, precision = compute_metrics(target_movie_rating.cpu().detach().numpy(),
#                                                                    out.cpu().detach().numpy())
#             train_auc.append(auc)
#             train_f1.append(f1)
#             train_acc.append(acc)
#             train_recall.append(recall)
#             train_precision.append(precision)
#
#         epoch_end_time = time.time()
#
#         print(
#             "train epoch : {}\tloss : {:.4f}\tauc : {:.4f}\tf1: {:.4f}\tacc :{:.4f}\trecall :{:.4f}\tprecision :{:.4f}\tspend : {:.4f} h".format(epoch,
#                 sum(train_losses) / len(train_losses), sum(train_auc) / len(train_auc), sum(train_f1) / len(train_f1),
#                 sum(train_acc) / len(train_acc), sum(train_recall) / len(train_recall),
#                 sum(train_precision) / len(train_precision), (epoch_start_time - start_time) / 60 / 60))
#
#         model.eval()
#         val_auc, val_losses, val_f1, val_acc, val_recall, val_precision = [], [], [], [], [], []
#
#         for datas in val_data:
#
#             datas = [data.to(device) for data in datas]
#             with torch.no_grad():
#                 out, target_movie_rating = model(datas)
#                 out = out.flatten()
#
#                 target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
#                 target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1
#
#                 loss = criterion(out, target_movie_rating)
#                 val_losses.append(loss.item())
#
#                 auc, f1, acc, recall, precision = compute_metrics(target_movie_rating.cpu().detach().numpy(),
#                                                                        out.cpu().detach().numpy())
#
#                 val_auc.append(auc)
#                 val_f1.append(f1)
#                 val_acc.append(acc)
#                 val_recall.append(recall)
#                 val_precision.append(precision)
#
#
#         print("test \tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(sum(val_losses) / len(val_losses), sum(val_auc) / len(val_auc), sum(val_f1) / len(val_f1), sum(val_acc) / len(val_acc), sum(val_recall) / len(val_recall),
#                        sum(val_precision) / len(val_precision)))
#
#         result_logging(epoch, sum(val_losses) / len(val_losses), sum(val_auc) / len(val_auc), sum(val_f1) / len(val_f1), sum(val_acc) / len(val_acc), sum(val_recall) / len(val_recall),
#                        sum(val_precision) / len(val_precision), running_time, hyper_param, writer)
#
#         # save best model
#         best_model_path = './' + running_time + "/" + hyper_param + "/" + '/logs/best_model/'
#         if not os.path.exists(best_model_path):
#             os.makedirs(best_model_path)
#         if (sum(val_auc) / len(val_auc)) > best_auc:
#             best_auc = sum(val_auc) / len(val_auc)
#             best_model = model.state_dict()
#             torch.save(best_model, best_model_path + 'best_model.pth')
#
#     print("Finished Training")
#
#     model.load_state_dict(torch.load(best_model_path + 'best_model.pth'))
#     model.eval()
#
#     test_auc, test_losses, test_f1, test_acc, test_recall, test_precision = [], [], [], [], [], []
#     for datas in test_data:
#         datas = [data.to(device) for data in datas]
#         with torch.no_grad():
#             out, target_movie_rating = model(datas)
#             out = out.flatten()
#
#             target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
#             target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1
#
#             loss = criterion(out, target_movie_rating)
#             test_losses.append(loss.item())
#
#             auc, f1, acc, recall, precision = compute_metrics(target_movie_rating.cpu().detach().numpy(),
#                                                               out.cpu().detach().numpy())
#
#             test_auc.append(auc)
#             test_f1.append(f1)
#             test_acc.append(acc)
#             test_recall.append(recall)
#             test_precision.append(precision)
#     print("test \tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(
#         sum(test_losses) / len(test_losses), sum(test_auc) / len(test_auc), sum(test_f1) / len(test_f1),
#         sum(test_acc) / len(test_acc), sum(test_recall) / len(test_recall),
#         sum(test_precision) / len(test_precision)))
#
#     # writer.add_scalar(tag="loss/test", scalar_value=sum(test_losses) / len(test_losses), global_step=epoch)
#     # writer.add_scalar(tag="auc/test", scalar_value=sum(test_auc) / len(test_auc), global_step=epoch)
#     # writer.add_scalar(tag="f1/test", scalar_value=sum(test_f1) / len(test_f1), global_step=epoch)
#     # writer.add_scalar(tag="acc/test", scalar_value=sum(test_acc) / len(test_acc), global_step=epoch)
#     # writer.add_scalar(tag="recall/test", scalar_value=sum(test_recall) / len(test_recall), global_step=epoch)
#     # writer.add_scalar(tag="precision/test", scalar_value=sum(test_precision) / len(test_precision), global_step=epoch)
#
#     writer.close()
#
#
#
# if __name__ == '__main__':
#     args = parser()
#     train(args)
#
#
#
#
#
#
#
#
#
#
