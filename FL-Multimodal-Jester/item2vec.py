import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
import random

# "https://github.com/AmazingDD/item2vec-pytorch"

data_path = "/huanggx/0002code/data/"

class Item2Vec(nn.Module):
    def __init__(self):
        super(Item2Vec, self).__init__()
        self.movies = pd.read_csv(data_path + "/movies.csv", sep=",")
        self.args = {
            'context_window': 2,
            'vocabulary_size': int(self.movies['movie_id'].max()) + 1,
            'rho': 1e-5,  # threshold to discard word in a sequence
            'batch_size': 256,
            'embedding_dim': 100,
            'epochs': 20,
            'learning_rate': 0.01,
            'UBI': 100,
            'n_users': 50,
            '_balance': False,
        }
        self.shared_embedding = nn.Embedding(self.args['vocabulary_size'], self.args['embedding_dim'])
        self.lr = self.args['learning_rate']
        self.epochs = self.args['epochs']
        self.out_act = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)




    def forward(self, input_data):
        target_i, context_j, label = input_data
        target_emb = self.shared_embedding(target_i) # batch_size * embedding_size
        context_emb = self.shared_embedding(context_j) # batch_size * embedding_size
        output = torch.sum(target_emb * context_emb, dim=1)
        output = self.out_act(output)

        return output, label.float()

    def get_data(self):
        train_dataset = get_data(self.args, data_path + "/new_dataset4/train_data.csv", if_train=True)
        val_dataset = get_data(self.args, data_path + "/new_dataset4/val_data.csv", if_train=False)
        test_dataset = get_data(self.args, data_path + "/new_dataset4/test_data.csv", if_train=False)

        ratings = pd.read_csv(data_path + "/ratings.csv", sep=",")
        n_u = ratings['user_id'].nunique()
        shuffle_l = shuffle_list(n_u, self.args['n_users'], self.args['UBI'])  # 包含0索引，6040个用户，50份，100最小数（随机取）
        print(shuffle_l)

        local_train_datasets = dataset_split(train_dataset, self.args['n_users'], balance=self.args['_balance'],
                                             path=data_path + "/new_dataset4/train_data.csv", shuffle_l=shuffle_l)
        local_val_datasets = dataset_split(val_dataset, self.args['n_users'], balance=self.args['_balance'],
                                           path=data_path + "/new_dataset4/val_data.csv", shuffle_l=shuffle_l)
        local_test_datasets = dataset_split(test_dataset, self.args['n_users'], balance=self.args['_balance'],
                                            path=data_path + "/new_dataset4/test_data.csv", shuffle_l=shuffle_l)


        return local_train_datasets, local_val_datasets, local_test_datasets

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

    list2 = generate_sorted_list_with_sum(1, amount-35, num-1, amount-35)
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


# class Item2Vec(nn.Module):
#     def __init__(self, args):
#         super(Item2Vec, self).__init__()
#         self.shared_embedding = nn.Embedding(args['vocabulary_size'], args['embedding_dim'])
#         self.lr = args['learning_rate']
#         self.epochs = args['epochs']
#         self.out_act = nn.Sigmoid()
#
#     def forward(self, target_i, context_j):
#         target_emb = self.shared_embedding(target_i) # batch_size * embedding_size
#         context_emb = self.shared_embedding(context_j) # batch_size * embedding_size
#         output = torch.sum(target_emb * context_emb, dim=1)
#         output = self.out_act(output)
#
#         return output.view(-1)
#
#     def fit(self, train_loader, val_dataloader):
#         if torch.cuda.is_available():
#             self.cuda()
#         else:
#             self.cpu()
#         optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         criterion = nn.BCEWithLogitsLoss(reduction='mean')
#
#         last_loss = 0.
#         for epoch in range(1, self.epochs + 1):
#             self.train()
#             current_loss = 0.
#             for target_i, context_j, label in train_loader:
#                 if torch.cuda.is_available():
#                     target_i = target_i.cuda()
#                     context_j = context_j.cuda()
#                     label = label.cuda()
#                 else:
#                     target_i = target_i.cpu()
#                     context_j = context_j.cpu()
#                     label = label.cpu()
#                 self.zero_grad()
#                 prediction = self.forward(target_i, context_j)
#                 loss = criterion(prediction, label)
#                 if torch.isnan(loss):
#                     raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the model')
#                 loss.backward()
#                 optimizer.step()
#                 current_loss += loss.item()
#             print(f'[Epoch {epoch:03d}] - training loss={current_loss:.4f}')
#             delta_loss = float(current_loss - last_loss)
#             if (abs(delta_loss) < 1e-5) and self.early_stop:
#                 print('Satisfy early stop mechanism')
#                 break
#             else:
#                 last_loss = current_loss
#
#
#             # val
#             self.eval()
#             for target_i, context_j, label in val_dataloader:
#                 with torch.no_grad():
#                     if torch.cuda.is_available():
#                         target_i = target_i.cuda()
#                         context_j = context_j.cuda()
#                         label = label.cuda()
#                     else:
#                         target_i = target_i.cpu()
#                         context_j = context_j.cpu()
#                         label = label.cpu()
#                     prediction = self.forward(target_i, context_j)
#                     loss = criterion(prediction, label)
#
#
#
#
#     def predict(self):
#         pass

# class MovieDataset(Dataset):
#     def __init__(self, ratings_file, test=False):
#         self.ratings_frame = pd.read_csv(ratings_file, delimiter=",")
#         self.test = test
#
#     def __len__(self):
#         return len(self.ratings_frame)
#
#     def __getitem__(self, idx):
#         data = self.ratings_frame.iloc[idx]
#         user_id = data.user_id
#
#         movie_history = eval(data.sequence_movie_ids)  # if sequence_movie_ids = 3186,1721,1270,1022,2340,1836,3408,1207; movie_history = (3186, 1721, 1270, 1022, 2340, 1836, 3408, 1207)
#         movie_history_ratings = eval(data.sequence_ratings)
#         movie_id = movie_history[-1:][0]  # movie_history is tuple, [0] takes the first element
#         rating = movie_history_ratings[-1:][0]
#
#         return user_id, movie_id, rating

def get_data(args, path, if_train=False):


    data = pd.read_csv(path, delimiter=",")
    data['item'] = data.apply(lambda x: eval(x['sequence_movie_ids'])[-1:][0], axis=1)
    data['rating'] = data.apply(lambda x: eval(x['sequence_ratings'])[-1:][0], axis=1)
    data_seqs = data.groupby('user_id')['item'].agg(list)
    word_frequency = data['item'].value_counts()
    prob_discard = 1 - np.sqrt(args['rho'] / word_frequency)
    sgns_samples = sgns_sample_generator(data_seqs, args['vocabulary_size'], args['context_window'], prob_discard)
    item2vec_dataset = Item2VecDataset(sgns_samples)
    # if if_train:
    #     data_loader = DataLoader(item2vec_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    # else:
    #     data_loader = DataLoader(item2vec_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)



    return item2vec_dataset


def choose_with_prob(p_discard):
    p = np.random.uniform(low=0.0, high=1.0)
    return False if p < p_discard else True

def sgns_sample_generator(train_seqs, vocabulary_size, context_window, prob_discard, discard=False):
    sgns_samples = []
    for seq in train_seqs:
        if discard:
            seq = [w for w in seq if choose_with_prob(prob_discard[w])]
        for i in range(len(seq)):
            target = seq[i]
            # generate positive sample
            context_list = []
            j = i - context_window
            while j <= i + context_window and j < len(seq):
                if j >= 0 and j != i:
                    context_list.append(seq[j])
                    sgns_samples.append([(target, seq[j]), 5])
                j += 1
            # generate negative sample
            for _ in range(len(context_list)):
                neg_idx = random.randrange(0, vocabulary_size)
                while neg_idx in context_list:
                    neg_idx = random.randrange(0, vocabulary_size)
                sgns_samples.append([(target, neg_idx), 0])
    return sgns_samples

class Item2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        label = self.data[index][1]
        xi = self.data[index][0][0]
        xj = self.data[index][0][1]
        label = torch.tensor(label, dtype=torch.float32)
        xi = torch.tensor(xi, dtype=torch.long)
        xj = torch.tensor(xj, dtype=torch.long)

        return xi, xj, label

    def __len__(self):
        return len(self.data)

def main():
    movies = pd.read_csv(data_path + "/movies.csv", sep=",")
    args = {
        'context_window': 2,
        'vocabulary_size': int(movies['movie_id'].max()) + 1,
        'rho': 1e-5,  # threshold to discard word in a sequence
        'batch_size': 256,
        'embedding_dim': 100,
        'epochs': 20,
        'learning_rate': 0.001,
    }
    # train_dataset = MovieDataset(data_path + "/new_dataset4/train_data.csv")
    # val_dataset = MovieDataset(data_path + "/new_dataset4/val_data.csv")
    # test_dataset = MovieDataset(data_path + "/new_dataset4/test_data.csv")

    # train_data, val_data, test_data = get_data(args['batch_size'])

    # train_ = pd.read_csv(data_path + "/new_dataset4/train_data.csv", delimiter=",")
    # train_['item'] = train_.apply(lambda x: eval(x['sequence_movie_ids'])[-1:][0], axis=1)
    # train_['rating'] = train_.apply(lambda x: eval(x['sequence_ratings'])[-1:][0], axis=1)
    # train_seqs = train_.groupby('user_id')['item'].agg(list)
    # word_frequency = train_['item'].value_counts()
    # prob_discard = 1 - np.sqrt(args['rho'] / word_frequency)
    # sgns_samples = sgns_sample_generator(train_seqs, args['vocabulary_size'], args['context_window'], prob_discard)
    # item2vec_dataset = Item2VecDataset(sgns_samples)
    # train_loader = DataLoader(item2vec_dataset, batch_size=args['batch_size'], shuffle=True)

    train_dataloader = get_data(args, data_path + "/new_dataset4/train_data.csv", if_train=True)
    val_dataloader = get_data(args, data_path + "/new_dataset4/val_data.csv", if_train=False)
    test_dataloader = get_data(args, data_path + "/new_dataset4/test_data.csv", if_train=False)


    model = Item2Vec(args)
    model.fit(train_dataloader, val_dataloader)




if __name__ == '__main__':
    main()





