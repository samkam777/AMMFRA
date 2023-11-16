import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class MFB(nn.Module):
    def __init__(self, img_feature_size, ques_feat_size, MFB_K = 5, MFB_O = 1000, is_first=True):
        super().__init__()
        self.is_first = is_first
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.proj_i = nn.Linear(img_feature_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)
        # self.init_weight()

    # def init_weight(self):
    #     nn.init.xavier_uniform_(self.proj_i)
    #     nn.init.xavier_uniform_(self.proj_q)

    def forward(self, img_feat, ques_feat, exp_in=1, **kwargs):
        '''
        :param img_feat:     img_feat.size() - > (batch_size, img_feat_size)
        :param ques_feat:    ques_feat.size() - > (batch_size, ques_feat_size)
        :param z:            z.size() - > (batch_size, MFB_O)
        :param exp_out:      exp_out.size() - > (batch_size, MFB_K * MFB_O)
        :return:
        '''

        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)        # (batch_size, K*O)
        ques_feat = self.proj_q(ques_feat)      # (batch_size, K*O)
        exp_out = img_feat * ques_feat          # (batch_size, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)    # (batch_size, K*O)
        z = self.pool(exp_out) * self.MFB_K     # (batch_size, O)
        z = torch.sqrt(F.relu(z)) * torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1)) # (batch_size, O)
        z = z.view(batch_size, self.MFB_O)      # (batch_size, O)

        return z, exp_out

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

        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector), dim=1)     # 128, 1


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
                                               int(math.sqrt(self.users.user_id.max())) + 1)  # Embedding(6041, 78)
        self.embeddings_user_sex = nn.Embedding(len(self.users.sex.unique()),
                                                int(math.sqrt(len(self.users.sex.unique()))))  # Embedding(2, 1)
        self.embeddings_age_group = nn.Embedding(len(self.users.age_group.unique()),
                                                 int(math.sqrt(len(self.users.age_group.unique()))))  # Embedding(7, 2)
        self.embeddings_user_occupation = nn.Embedding(len(self.users.occupation.unique()),
                                                       int(math.sqrt(
                                                           len(self.users.occupation.unique()))))  # Embedding(21, 4)

        # movie
        movie_embedding_dims = int(math.sqrt(self.movies.movie_id.max())) + 1  # 63
        self.embeddings_movie_id = nn.Embedding(int(self.movies.movie_id.max()) + 1,
                                                movie_embedding_dims)  # Embedding(3953, 63)
        # Create a vector lookup for movie genres.
        genres_embedding_dims = 73  # ont-hot码73维

        self.transformerlayer = nn.TransformerEncoderLayer(136, 4, dropout=0.2)
        # self.transformerlayer = nn.TransformerEncoderLayer(90, 3, dropout=0.2)

        self.linear = nn.Sequential(  # nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1), )

        self.img_linear = nn.Sequential(nn.Linear(1024, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256), )

        self.txt_linear = nn.Sequential(nn.Linear(768, 512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512, 256), )

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

        # self.MFB = MFB(1024, 768)

        self.attention_layer = AdditiveAttention(256, 256)

    def init_weight(self):
        nn.init.normal_(self.embeddings_user_id.weight, std=0.01)
        nn.init.normal_(self.embeddings_user_sex.weight, std=0.01)
        nn.init.normal_(self.embeddings_age_group.weight, std=0.01)
        nn.init.normal_(self.embeddings_user_occupation.weight, std=0.01)
        nn.init.normal_(self.embeddings_movie_id.weight, std=0.01)

        for name, param in self.linear.named_parameters():
            nn.init.normal_(param, mean=0, std=0.01)


    def encode_input(self, inputs):
        # len: 1, 7, 1, 7, 1, 1, 1, 1

        user_id, movie_history, target_movie_id, movie_history_ratings, target_movie_rating, sex, age_group, occupation, movie_history_genres, target_movie_genres, target_movie_id_image_feature, target_movie_id_txt_feature= inputs


        target_movie_emb = self.embeddings_movie_id(target_movie_id)        # Batchsize dim   # torch.size([128, 63])
        user_id = self.embeddings_user_id(user_id)                          # Batchsize dim   # torch.size([128, 78])
        user_features = torch.cat((user_id, target_movie_emb), 1)           # Batchsize dim   # torch.size([128, 141])

        # behavior
        movie_history_id_emb = self.embeddings_movie_id(movie_history)      # 128, 3, 63
        movie_history_emb = torch.cat((movie_history_id_emb, movie_history_genres), dim=2) # 128, 3, 63+73 = 136

        target_movie_emb = torch.cat((target_movie_emb, target_movie_genres), dim=1)# 128, 63+73=136
        target_movie_emb = target_movie_emb.unsqueeze(1)  # 128, 1, 136
        # target_movie_genres = target_movie_genres.unsqueeze(1)  # 128, 1, 73
        # print("movie_history_genres.shape: {}".format(movie_history_genres.shape))
        # print("target_movie_genres.shape: {}".format(target_movie_genres.shape))
        transformer_features = torch.cat((movie_history_emb, target_movie_emb), dim=1) # 128, 4, 136


        return transformer_features, user_features, target_movie_rating.float(), target_movie_id_image_feature, target_movie_id_txt_feature


    def forward(self, x):
        # transformer_features, user_features, target_movie_rating, image_transformer_features = self.encode_input(x)
        transformer_features, user_features, target_movie_rating, target_movie_id_image_feature, target_movie_id_txt_feature = self.encode_input(
            x)
        # output = 0

        transformer_output = self.transformerlayer(transformer_features)
        # print("transformer_output : {}".format(transformer_output.shape))               # transformer_output : torch.Size([128, 4, 63])

        transformer_output = torch.flatten(transformer_output, start_dim=1)
        # print("transformer_output : {}".format(transformer_output.shape))               # transformer_output : torch.Size([128, 4*136=544])
        transformer_output = self.transformer_linear(transformer_output)
        transformer_output = transformer_output.unsqueeze(1)
        transformer_output = self.attention_layer(transformer_output)

        # Concat with other features
        # features = torch.cat((transformer_output, user_features), dim=1)
        # print("features : {}".format(features.shape))                                   # features : torch.Size([128, 544+141=685])

        img_features = self.img_linear(target_movie_id_image_feature)  # 128, 256
        img_features = img_features.unsqueeze(1)  # 128, 1, 256
        img_features = self.attention_layer(img_features)  # 128, 256

        txt_features = self.txt_linear(target_movie_id_txt_feature)  # 128, 256
        txt_features = txt_features.unsqueeze(1)
        txt_features = self.attention_layer(txt_features)  # 128, 256

        user_features = self.user_linear(user_features)
        user_features = user_features.unsqueeze(1)
        user_features = self.attention_layer(user_features)

        # MFB_out, _ = self.MFB(target_movie_id_image_feature, target_movie_id_txt_feature)   # 128, 1000

        features = torch.cat((user_features, transformer_output, img_features, txt_features), dim=1)  # 128, 256*4=1024

        output = self.linear(features)

        output = self.logistic(output)

        return output, target_movie_rating








