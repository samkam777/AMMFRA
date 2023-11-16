import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torch
import torch.nn as nn
import torch.optim as optim

"https://github.com/AmazingDD/item2vec-pytorch"

data_path = "/home/jadeting/samkam/code/dataset/data/"

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
            'learning_rate': 0.001,
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
        train_dataloader = get_data(self.args, data_path + "/new_dataset4/train_data.csv", if_train=True)
        val_dataloader = get_data(self.args, data_path + "/new_dataset4/val_data.csv", if_train=False)
        test_dataloader = get_data(self.args, data_path + "/new_dataset4/test_data.csv", if_train=False)

        return train_dataloader, val_dataloader, test_dataloader






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

class MovieDataset(Dataset):
    def __init__(self, ratings_file, test=False):
        self.ratings_frame = pd.read_csv(ratings_file, delimiter=",")
        self.test = test

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        movie_history = eval(data.sequence_movie_ids)  # if sequence_movie_ids = 3186,1721,1270,1022,2340,1836,3408,1207; movie_history = (3186, 1721, 1270, 1022, 2340, 1836, 3408, 1207)
        movie_history_ratings = eval(data.sequence_ratings)
        movie_id = movie_history[-1:][0]  # movie_history is tuple, [0] takes the first element
        rating = movie_history_ratings[-1:][0]

        return user_id, movie_id, rating

def get_data(args, path, if_train=False):


    data = pd.read_csv(path, delimiter=",")
    data['item'] = data.apply(lambda x: eval(x['sequence_movie_ids'])[-1:][0], axis=1)
    data['rating'] = data.apply(lambda x: eval(x['sequence_ratings'])[-1:][0], axis=1)
    data_seqs = data.groupby('user_id')['item'].agg(list)
    word_frequency = data['item'].value_counts()
    prob_discard = 1 - np.sqrt(args['rho'] / word_frequency)
    sgns_samples = sgns_sample_generator(data_seqs, args['vocabulary_size'], args['context_window'], prob_discard)
    item2vec_dataset = Item2VecDataset(sgns_samples)
    if if_train:
        data_loader = DataLoader(item2vec_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
    else:
        data_loader = DataLoader(item2vec_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)

    return data_loader


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





