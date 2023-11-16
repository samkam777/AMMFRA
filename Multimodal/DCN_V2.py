import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, dropout_rate):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i+1]) for i in range(len(self.hidden_units) - 1)
        ])

        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        inputs = x
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs

class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=42):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[0])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_1 = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                x1_w = torch.tensordot(x_1, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, x1_w)
                x_1 = dot_ + self.bias[i] + x_1
            else:
                x1_w = torch.tensordot(self.kernels[i], x_1)
                dot_ = x1_w + self.bias[i]
                x_1 = x_0 * dot_ + x_1
        x_1 = torch.squeeze(x_1, dim=2)
        return x_1


class DCNV2(nn.Module):
    def __init__(self, users, movies, device, dnn_hidden_units=(128,128,), cross_param='vector', args=None, dropout = 0.5, l2_reg=0.00001):
        super(DCNV2, self).__init__()
        self.users = users
        self.movies = movies
        self.device = device
        self.args = args

        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = 2
        self.cross_param = cross_param
        self.dropout = nn.Dropout(dropout)
        self.l2_reg = l2_reg
        self.act = nn.ReLU()






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

        inputs_dim = int(math.sqrt(self.users.user_id.max())) + 1 + int(math.sqrt(len(self.users.sex.unique()))) + int(
            math.sqrt(len(self.users.age_group.unique()))) + int(
            math.sqrt(len(self.users.occupation.unique()))) + movie_embedding_dims

        self.dnn = DNN(inputs_dim, self.dnn_hidden_units, 0.5)
        self.crossnet = CrossNet(inputs_dim, layer_num=self.cross_num, parameterization=self.cross_param)
        self.dnn_linear = nn.Linear(inputs_dim+dnn_hidden_units[-1], 1, bias=False)

        dnn_hidden_units = [inputs_dim] + list(dnn_hidden_units) + [1]
        # print(dnn_hidden_units)       # 148, 128, 128, 1

        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i+1]) for i in range(len(dnn_hidden_units) - 1)
        ])

        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)






    def forward(self, x):

        user_id, _, target_movie_id, _, target_movie_rating, sex, age_group, occupation, _, _, _, _ = x
        batch_size = user_id.size(0)

        sex_emb = self.embeddings_user_sex(sex)
        age_emb = self.embeddings_age_group(age_group)
        occupation_emb = self.embeddings_user_occupation(occupation)

        target_movie_emb = self.embeddings_movie_id(target_movie_id)  # Batchsize dim   # torch.size([128, 63])
        user_id = self.embeddings_user_id(user_id)  # Batchsize dim   # torch.size([128, 78])
        features = torch.cat((user_id, target_movie_emb, sex_emb, age_emb, occupation_emb), 1)  # Batchsize dim   # torch.size([128, 141])

        logit = features

        # print("features : {}".format(features.shape))

        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc

        deep_out = self.dnn(features)
        cross_out = self.crossnet(features)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)

        logit_ = logit + self.dnn_linear(stack_out)
        # print("logit:{}".format(logit_))
        y_pred = torch.sigmoid(logit_)
        print("y_pred:{}".format(y_pred))

        return y_pred, target_movie_rating.float()






