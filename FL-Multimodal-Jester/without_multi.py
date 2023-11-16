import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class BST_without(nn.Module):
    def __init__(self, users, ratings, movies, genres, device, args=None):
        super(BST_without, self).__init__()
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

        self.linear = nn.Sequential(  # nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(141, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1), )


        self.logistic = nn.Sigmoid()


    def encode_input(self, inputs):

        user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature = inputs

        target_movie_emb = self.embeddings_movie_id(target_movie_id)        # Batchsize dim   # torch.size([128, 63])
        user_id = self.embeddings_user_id(user_id)                          # Batchsize dim   # torch.size([128, 78])
        user_features = torch.cat((user_id, target_movie_emb), 1)           # Batchsize dim   # torch.size([128, 141])

        return user_features, target_movie_rating.float()

    def forward(self, x):

        user_features, target_movie_rating = self.encode_input(x)

        output = self.linear(user_features)

        output = self.logistic(output)

        return output, target_movie_rating