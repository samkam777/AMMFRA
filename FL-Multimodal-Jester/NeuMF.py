import torch
import torch.nn as nn

'''


'''

class NeuMF(nn.Module):
    def __init__(self, users, ratings, movies, genres, device):
        super(NeuMF, self).__init__()
        # self.num_users = num_users
        # self.num_items = num_items
        self.users = users
        self.ratings = ratings
        self.movies = movies
        self.genres = genres
        self.factor_num_mf = 32
        self.layers = [64, 32, 16, 8]
        self.factor_num_mlp = int(self.layers[0] / 2)

        self.dropout = 0.2

        self.embedding_user_mlp = nn.Embedding(num_embeddings=int(self.users.user_id.max()) + 1, embedding_dim=self.factor_num_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=int(self.movies.movie_id.max()) + 1, embedding_dim=self.factor_num_mlp)

        self.embedding_user_mf = nn.Embedding(num_embeddings=int(self.users.user_id.max()) + 1, embedding_dim=self.factor_num_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=int(self.movies.movie_id.max()) + 1, embedding_dim=self.factor_num_mf)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        # print(self.fc_layers)

        self.affine_output = nn.Linear(in_features=self.layers[-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

        self.loss = nn.BCELoss().to(device)

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        # user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature
        user_indices, movie_history, item_indices, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature = inputs

        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        # print("mlp_vector:{}".format(mlp_vector.shape))     # batch_size, 64

        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)  # 两矩阵的内积

        # 全连接网络层
        for idx, _ in enumerate(range(len(self.fc_layers))):
            # print(self.fc_layers[idx])
            # print("mlp_vector:{}".format(mlp_vector.shape))     # batch_size, 64
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)  # sigmoid
        return rating, target_movie_rating.float()

