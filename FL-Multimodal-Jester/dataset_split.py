import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import random
from parser import parser

data_path = "/huanggx/0002code/data/Jester/"

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

class MovieDataset(Dataset):
    def __init__(self, ratings_file, test=False):
        self.ratings_frame = pd.read_csv(ratings_file, delimiter=",")
        self.test = test
        # self.img_feature_df = pd.read_csv(data_path + "/img_feature/img_feature_1024_V2.csv")
        # self.genres_feature_df = pd.read_csv(data_path + "/movies_genres.csv", sep=",")
        # # print(self.genres_feature_df)
        # # self.txt_feature_tensor = torch.load("/home/jadeting/samkam/code/dataset/data/bert_feature/bert_extract.pth")
        self.txt_feature_df = pd.read_csv(data_path + "/Jester_txt_feature.csv")

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


        # # movie_history_image_feature = image_feature(self.img_feature_df, movie_history, True)
        # target_movie_id_image_feature = image_feature(self.img_feature_df, target_movie_id, False)
        # # print(movie_history_image_feature)
        #
        # # target movie txt bert feature
        target_movie_id_txt_feature = txt_feature(self.txt_feature_df, target_movie_id, False)


        # movie_history_genres = genres_feature(self.genres_feature_df, movie_history, True)
        # target_movie_genres = genres_feature(self.genres_feature_df, target_movie_id, False)
        # # print("target_movie_genres : {}".format(target_movie_genres))

        # sex = data.sex
        # age_group = data.age_group
        # occupation = data.occupation

        return user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, target_movie_id_txt_feature

def get_data(args):


    # users = pd.read_csv(data_path + "/users.csv", sep=",")
    # ratings = pd.read_csv(data_path + "/ratings.csv", sep=",")
    # movies = pd.read_csv(data_path + "/movies.csv", sep=",")
    # genres = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    #           "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    #
    # for genre in genres:
    #     movies[genre] = movies["genres"].apply(lambda values: int(genre in values.split("|")))

    # genre_vectors = movies[genres].to_numpy()

    # train_dataset = MovieDataset(data_path + "/train_data.csv")
    # val_dataset = MovieDataset(data_path + "/test_data.csv")
    # test_dataset = MovieDataset(data_path + "/test_data.csv")
    user = pd.read_csv(data_path + "/train_data_new3.csv")

    n_u = user['user_id'].nunique()


    shuffle_l = shuffle_list(n_u, args.n_users, args.UBI)  # 包含0索引，6040个用户，50份，100最小数（随机取）
    print(shuffle_l)
    train_dataset = MovieDataset(data_path + "/train_data_new3.csv")
    val_dataset = MovieDataset(data_path + "/val_data_new3.csv")
    test_dataset = MovieDataset(data_path + "/test_data_new3.csv")

    users = pd.read_csv(data_path + "/train_data_new3.csv")
    movies = pd.read_csv(data_path + "/new_jester_datasetV1.csv")


    local_train_datasets = dataset_split(train_dataset, args.n_users, balance=args._balance, path=data_path + "/train_data_new3.csv", shuffle_l=shuffle_l)
    local_val_datasets = dataset_split(val_dataset, args.n_users, balance=args._balance, path=data_path + "/val_data_new3.csv", shuffle_l=shuffle_l)
    local_test_datasets = dataset_split(test_dataset, args.n_users, balance=args._balance, path=data_path + "/test_data_new3.csv", shuffle_l=shuffle_l)

    # train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                          num_workers=4)
    # val_data = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    #                                        num_workers=4)
    # test_data = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
    #                                         num_workers=4)

    # train_data_sub = torch.utils.data.DataLoader(local_train_datasets[0], batch_size=batch_size, shuffle=True,
    #                                             num_workers=4)
    # val_data = torch.utils.data.DataLoader(local_val_datasets, batch_size=batch_size, shuffle=False,
    #                                             num_workers=4)
    # test_data = torch.utils.data.DataLoader(local_test_datasets, batch_size=batch_size, shuffle=False,
    #                                             num_workers=4)
    # print(train_data_sub)
    #
    # for item in train_data_sub:
    #     # aa,_,_,_,_,_,_ = item
    #     print(len(item))


    return users, movies, local_train_datasets, local_val_datasets, local_test_datasets


class DataSubset(Subset):
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x = self.dataset[self.indices[idx]]
        if self.subset_transform:
            x = self.subset_transform(x)
        return x

    def __len__(self):
        return len(self.indices)

def dataset_split(dataset, n_workers, balance, path, shuffle_l = None):
    # n_samples = len(dataset)
    # shuffle_l = shuffle_list(n_samples, n_workers, 128)
    # n_samples_per_workers = n_samples // n_workers

    # train_data_df = pd.read_csv(ratings_file, delimiter=",")

    ##########

    data_df = pd.read_csv(path, sep=",")
    # print(train_data_df.head())
    n_user = data_df['user_id'].nunique()     # 计算用户数量
    # print(n_user)
    userid_g = data_df.groupby('user_id')     # 按user_id分组
    # print(userid_g)
    # print(userid_g.tail(1))                         # user id的最后一行数据
    data_index = userid_g.tail(1).index.tolist()
    data_index.insert(0, 0)                         # 在列首添加了一个0，作为开头索引
    # print(userid_g.tail(1).index)                   # user id的最后一行数据的索引数组
    # print(len(userid_g.tail(1).index))              # 索引数组的长度，应该与n_user一致

    # shuffle_l = shuffle_list(n_user, n_workers, 100)    # 包含0索引
    n_balance = n_user // n_workers
    # last_sample_data_index = data_index[-1]


    local_datasets = []

    if balance:
        for w_id in range(n_workers):
            if w_id < n_workers - 1:
                # local_datasets.append(DataSubset(dataset, range(shuffle_l[w_id], shuffle_l[w_id + 1])))
                local_datasets.append(DataSubset(dataset, range(data_index[w_id * n_balance], data_index[(w_id + 1) * n_balance])))
            else:
                # local_datasets.append(DataSubset(dataset, range(shuffle_l[-1], n_samples)))
                local_datasets.append(DataSubset(dataset, range(data_index[w_id * n_balance], data_index[-1])))
    else:
        for w_id in range(n_workers):
            if w_id < n_workers - 1:
                # local_datasets.append(DataSubset(dataset, range(w_id * n_samples_per_workers, (w_id + 1) * n_samples_per_workers)))
                local_datasets.append(DataSubset(dataset, range(data_index[shuffle_l[w_id]], data_index[shuffle_l[w_id + 1]])))
            else:
                # local_datasets.append(DataSubset(dataset, range(w_id * n_samples_per_workers, n_samples)))
                local_datasets.append(DataSubset(dataset, range(data_index[shuffle_l[w_id]], data_index[-1])))

    return local_datasets


# amount 是总的数量，num 是要切分的份数。
def shuffle_list(amount, num, min_amount):
    '''
    得到类似这样的list
    list2 = [304, 708, 1341, 2985, 702]
    list3 = [304, 1012, 2353, 5338, 6040]
    '''
    # list1 = []
    # list1.append(0)
    # for i in range(0,num-1):
    #     a = random.randint(min_amount,amount)    # 生成 n-1 个随机节点
    #     # print("a:{}\t".format(a))
    #     list1.append(a)
    # list1.sort()                        # 节点排序
    # list1.append(amount)                # 设置第 n 个节点为amount，即总的数量
    #
    # list2 = []
    # for i in range(len(list1)):
    #     if i == 0:
    #         b = list1[i]                # 第一段长度为第 1 个节点 - 0
    #     else:
    #         b = list1[i] - list1[i-1]   # 其余段为第 n 个节点 - 第 n-1 个节点
    #     list2.append(b)

    list2 = generate_sorted_list_with_sum(1, amount-350, num-1, amount-350)
    # list2 = generate_sorted_list_with_sum(1, amount - 2500-150, num - 17, amount - 2500-150)
    #
    # list_low = [12, 22, 37, 50, 60, 71, 83, 97, 127, 135]
    # list_low = [3390 + d for d in list_low]
    # list2 = list2 + list_low
    # list_high = [4040, 4640, 5000, 5650, 6010]
    # list2 = list2 + list_high

    # list2 = [28, 55, 85, 128, 181, 234, 286, 345, 412, 484, 575, 662, 752, 845, 945, 1044, 1143, 1251, 1358, 1472, 1588, 1707, 1826, 1956, 2086, 2218, 2350, 2487, 2624, 2768, 2911, 3058, 3221, 3390, 3402, 3412, 3427, 3440, 3450, 3461, 3473, 3487, 3517, 3525, 4040, 4640, 5000, 5650, 6010]

    list2.insert(0, 0)

    list3 = []
    number = 0
    for i in range(num):
       number += list2[i]
       list3.append(number)

    return list3

def generate_sorted_list_with_sum(start, end, length, total_sum):
    if start > end:
        start, end = end, start
    if length > end - start + 1:
        raise ValueError("List length must be less than or equal to range size")
    mean = total_sum / length
    std = (total_sum / length) /  2
    lst = []
    while len(lst) < length:
        num = int(round(random.gauss(mean, std)))
        if num >= start and num <= end and num not in lst:
            lst.append(num)
    lst.sort()
    current_sum = sum(lst)
    while current_sum != total_sum:
        if current_sum < total_sum:
            idx = random.randint(0, length - 1)
            lst[idx] += 1
            current_sum += 1
        else:
            idx = random.randint(0, length - 1)
            if lst[idx] > start:
                lst[idx] -= 1
                current_sum -= 1
    return lst

if __name__ == '__main__':
    # batch_size = 128
    # users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data()
    # local_train_data = torch.utils.data.DataLoader(local_train_datasets[0], batch_size=batch_size, shuffle=True,
    #                                             num_workers=4)
    # print(len(local_train_datasets[0]))
    #
    # print(len(local_train_data)*128)    # client的数据量
    #
    # for data in local_train_data:
    #     print(len(data[0]))        # 128



    # ##### test 2
    # # print(shuffle_list(800000, 50, 100))
    # # print(len(shuffle_list(800000, 50, 100)))
    #
    # train_dataset = MovieDataset(data_path + "/new_dataset/train_data.csv")
    # val_dataset = MovieDataset(data_path + "/new_dataset/val_data.csv")
    # test_dataset = MovieDataset(data_path + "/new_dataset/test_data.csv")
    #
    # local_train_datasets = dataset_split(train_dataset, 50, balance=False, path=data_path + "/new_dataset/train_data.csv")
    # local_val_datasets = dataset_split(val_dataset, 50, balance=True, path=data_path + "/new_dataset/val_data.csv")
    # local_test_datasets = dataset_split(test_dataset, 50, balance=True, path=data_path + "/new_dataset/test_data.csv")
    # print(len(local_train_datasets))
    # print(len(local_val_datasets))
    # print(len(local_test_datasets))
    #
    # for data in local_train_datasets:
    #     print(len(data))
    #
    # for data in local_val_datasets:
    #     print(len(data))
    #
    # for data in local_test_datasets:
    #     print(len(data))


    # ######## test3
    # train_data_df = pd.read_csv(data_path + "/new_dataset/train_data.csv", sep=",")
    # print(train_data_df.head())
    # n_user = train_data_df['user_id'].nunique()
    # print(n_user)
    # userid_g = train_data_df.groupby('user_id')
    # print(userid_g)
    # print(userid_g.tail(1))     # 当前user id的最后一行数据
    # print(userid_g.tail(1).index)
    # # print(userid_g.tail(1).index.tolist())
    # userid_g_l = userid_g.tail(1).index.tolist()
    # userid_g_l.insert(0,0)
    # print(userid_g_l)
    # print(len(userid_g_l))
    # print(len(userid_g.tail(1).index))

    # ######## test4
    # train_data_df = pd.read_csv(data_path + "/new_dataset4/train_data.csv", sep=",")
    # val_data_df = pd.read_csv(data_path + "/new_dataset4/val_data.csv", sep=",")
    # test_data_df = pd.read_csv(data_path + "/new_dataset4/test_data.csv", sep=",")
    # n_train_user = train_data_df['user_id'].nunique()
    # n_val_user = val_data_df['user_id'].nunique()
    # n_test_user = test_data_df['user_id'].nunique()
    #
    # print("train :{}".format(n_train_user))
    # print("val :{}".format(n_val_user))
    # print("test :{}".format(n_test_user))


    ######## test5
    # shuffle_l = shuffle_list(6040, 50, 100)    # 包含0索引，6040个用户，50份，100最小数（随机取）
    # train_dataset = MovieDataset(data_path + "/new_dataset4/train_data.csv")
    # val_dataset = MovieDataset(data_path + "/new_dataset4/val_data.csv")
    # test_dataset = MovieDataset(data_path + "/new_dataset4/test_data.csv")
    #
    # local_train_datasets = dataset_split(train_dataset, 50, balance=False, path=data_path + "/new_dataset4/train_data.csv", shuffle_l=shuffle_l)
    # local_val_datasets = dataset_split(val_dataset, 50, balance=False, path=data_path + "/new_dataset4/val_data.csv", shuffle_l=shuffle_l)
    # local_test_datasets = dataset_split(test_dataset, 50, balance=False, path=data_path + "/new_dataset4/test_data.csv", shuffle_l=shuffle_l)
    # print(len(local_train_datasets))
    # print(len(local_val_datasets))
    # print(len(local_test_datasets))
    #
    # for data in local_train_datasets:
    #     print(len(data))
    #
    # for data in local_val_datasets:
    #     print(len(data))
    #
    # for data in local_test_datasets:
    #     print(len(data))

    args = parser()
    users, ratings, movies, genres, local_train_datasets, local_val_datasets, local_test_datasets = get_data(args)












