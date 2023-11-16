import torch
import os
import numpy as np
import copy
import torch.nn.functional as F
from logging_result import server_logging, global_model_logging


class Server:
    def __init__(self, device, args, train_dataset, val_dataset, test_dataset, model, running_time, hyper_param, writer, criterion):
        self.device = device
        self.global_epochs = args.epochs
        self.batch_size = args.batch_size
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.n_users = args.n_users
        self.n_select_users_rate = args.num_select_users_rate
        self.running_time = running_time
        self.hyper_param = hyper_param
        # DP
        self.if_DP = args.if_DP
        # self.if_poison = args.if_poison

        self.writer = writer

        # best auc
        self.pFedMe_global_best_auc = 0
        self.pFedMe_local_best_auc = 0
        self.fedavg_global_best_auc = 0

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def get_att_weight(self, user):
        att_weight = torch.tensor(0.).to(self.device)
        similarity = torch.tensor(0.).to(self.device)
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            att_weight += torch.norm(server_param - user_param, p=2)
            similarity += torch.cosine_similarity(server_param.view(-1), user_param.view(-1), dim=0)
        return att_weight, similarity

    def get_users_distance(self, user_self, user_others):
        distance = torch.tensor(0.).to(self.device)
        for self_param, others_param in zip(user_self.get_parameters(), user_others.get_parameters()):
            distance += torch.norm(self_param - others_param, p=2)
        return distance

################# weight parameters update #################
    # FedAvg aggregate
    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.n_train_data_samples
        # print("total_train:{}\t".format(total_train))
        ratio_list = []
        for user in self.selected_users:
            self.add_parameters(user, user.n_train_data_samples / total_train)
            ratio_list.append(user.n_train_data_samples / total_train)
        print("ratio:{} \t".format(ratio_list))

    # median attention pFed(用余弦相似性来度量) 2021.12.20
    def median_attention_persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))

        att_sim = []
        for user in self.selected_users:
            att_weight, similarity = self.get_att_weight(user)
            att_sim.append(similarity)
        print("similarity:{}\t".format(att_sim))
        att_sim_ = torch.Tensor(att_sim)
        # print("similarity:{}\t".format(att_sim_))
        min_att_sim_ = torch.min(att_sim_)
        max_att_sim_ = torch.max(att_sim_)
        median = self.get_median_data(att_sim_)
        print("median:{}\t".format(median))
        norm_att_sim_ = (att_sim_ - median) / (max_att_sim_ - min_att_sim_)
        norm_att_sim_ = torch.abs(norm_att_sim_)
        norm_att_sim_ = 1 - norm_att_sim_
        print("norm_att_sim_ before normalization: {}\t".format(norm_att_sim_))
        # norm_att_sim_ = norm_att_sim_ / norm_att_sim_.sum()
        norm_att_sim_ = F.softmax(norm_att_sim_)
        print("norm_att_sim_ after normalization: {}\t".format(norm_att_sim_))

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)  # 全局模型这里致0了

        for i, user in enumerate(self.selected_users):
            self.add_parameters(user, norm_att_sim_[i])

    def get_median_data(self, data):
        data = sorted(data)
        size = len(data)
        if size % 2 == 0:
            median = (data[size // 2] + data[size // 2 - 1]) / 2
        if size % 2 == 1:
            median = data[(size - 1) // 2]
        return median

    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        '''
        if (num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        # np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False)  # , p=pk)

############################################################


###################### server evaluate ######################
    # FedAvg
    def test(self):
        total_auc, total_losses, total_f1, total_acc, total_recall, total_precision = [], [], [], [], [], []
        for c in self.users:
            auc, losses, f1, acc, recall, precision = c.test()
            total_auc.append(auc)
            total_losses.append(losses)
            total_f1.append(f1)
            total_acc.append(acc)
            total_recall.append(recall)
            total_precision.append(precision)
        ids = [c.id for c in self.users]
        # print("len of total_HR:{}\t".format(len(total_auc)))

        return ids, total_auc, total_losses, total_f1, total_acc, total_recall, total_precision

    # global evaluation
    def global_test(self):
        total_auc, total_losses, total_f1, total_acc, total_recall, total_precision = [], [], [], [], [], []
        for c in self.users:
            auc, losses, f1, acc, recall, precision = c.global_test()
            total_auc.append(auc)
            total_losses.append(losses)
            total_f1.append(f1)
            total_acc.append(acc)
            total_recall.append(recall)
            total_precision.append(precision)
        ids = [c.id for c in self.users]
        # print("len of total_HR:{}\t".format(len(total_HR)))

        return ids, total_auc, total_losses, total_f1, total_acc, total_recall, total_precision

    def global_FedAPPLE_test(self):
        total_auc, total_losses, total_f1, total_acc, total_recall, total_precision = [], [], [], [], [], []
        for c in self.users:
            auc, losses, f1, acc, recall, precision = c.global_FedAPPLE_test()
            total_auc.append(auc)
            total_losses.append(losses)
            total_f1.append(f1)
            total_acc.append(acc)
            total_recall.append(recall)
            total_precision.append(precision)
        ids = [c.id for c in self.users]
        # print("len of total_HR:{}\t".format(len(total_HR)))

        return ids, total_auc, total_losses, total_f1, total_acc, total_recall, total_precision

    # pFedMe
    def test_persionalized_model(self):
        total_auc, total_losses, total_f1, total_acc, total_recall, total_precision = [], [], [], [], [], []

        for c in self.users:
            auc, losses, f1, acc, recall, precision = c.test_persionalized_model()
            total_auc.append(auc)
            total_losses.append(losses)
            total_f1.append(f1)
            total_acc.append(acc)
            total_recall.append(recall)
            total_precision.append(precision)

            # save best model
            best_model_path = './' + self.running_time + "/" + self.hyper_param + "/" + '/logs/user/user_' + str(c.id) + '/best_model/'
            if not os.path.exists(best_model_path):
                os.makedirs(best_model_path)
            if auc > c.pFedMe_local_best_auc:
                c.pFedMe_local_best_auc = auc
                best_model = c.persionalized_model_bar.state_dict()
                torch.save(best_model, best_model_path + "best_model.pth")


        ids = [c.id for c in self.users]
        # print("len of total_HR:{}\t".format(len(total_HR)))

        return ids, total_auc, total_losses, total_f1, total_acc, total_recall, total_precision

##########
    # FedAvg
    def FedAvg_global_evaluate(self, epoch):
        stats = self.global_test()
        auc = sum(stats[1]) / len(stats[1])
        loss = sum(stats[2]) / len(stats[2])
        f1 = sum(stats[3]) / len(stats[3])
        acc = sum(stats[4]) / len(stats[4])
        recall = sum(stats[5]) / len(stats[5])
        precision = sum(stats[6]) / len(stats[6])

        print("Evaluate global model\tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(loss, auc, f1, acc, recall, precision))
        global_model_logging(epoch, loss, auc, f1, acc, recall, precision, self.running_time, self.hyper_param, self.writer)

        # save best model
        best_model_path = './' + self.running_time + "/" + self.hyper_param + "/" + '/logs/global_model/best_model/'
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        if auc > self.pFedMe_global_best_auc:
            self.pFedMe_global_best_auc = auc
            best_model = self.model.state_dict()
            torch.save(best_model, best_model_path + 'best_model.pth')

    def FedAPPLE_global_evaluate(self, epoch):
        stats = self.global_FedAPPLE_test()
        auc = sum(stats[1]) / len(stats[1])
        loss = sum(stats[2]) / len(stats[2])
        f1 = sum(stats[3]) / len(stats[3])
        acc = sum(stats[4]) / len(stats[4])
        recall = sum(stats[5]) / len(stats[5])
        precision = sum(stats[6]) / len(stats[6])

        print("Evaluate global model\tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(loss, auc, f1, acc, recall, precision))
        global_model_logging(epoch, loss, auc, f1, acc, recall, precision, self.running_time, self.hyper_param, self.writer)

        # save best model
        best_model_path = './' + self.running_time + "/" + self.hyper_param + "/" + '/logs/global_model/best_model/'
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        if auc > self.pFedMe_global_best_auc:
            self.pFedMe_global_best_auc = auc
            best_model = self.model.state_dict()
            torch.save(best_model, best_model_path + 'best_model.pth')


    # FedAvg
    def evaluate(self, epoch, losses):
        stats = self.test()
        auc = sum(stats[1]) / len(stats[1])
        loss = sum(stats[2]) / len(stats[2])
        f1 = sum(stats[3]) / len(stats[3])
        acc = sum(stats[4]) / len(stats[4])
        recall = sum(stats[5]) / len(stats[5])
        precision = sum(stats[6]) / len(stats[6])

        print("Average server\tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(loss, auc, f1, acc, recall, precision))
        server_logging(epoch, loss, auc, f1, acc, recall, precision, self.running_time, self.hyper_param, self.writer)

    # pFedMe
    def pFedMe_global_evaluate(self, epoch):
        stats = self.global_test()
        auc = sum(stats[1]) / len(stats[1])
        loss = sum(stats[2]) / len(stats[2])
        f1 = sum(stats[3]) / len(stats[3])
        acc = sum(stats[4]) / len(stats[4])
        recall = sum(stats[5]) / len(stats[5])
        precision = sum(stats[6]) / len(stats[6])

        print("Evaluate global model\tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(loss, auc, f1, acc, recall, precision))
        global_model_logging(epoch, loss, auc, f1, acc, recall, precision, self.running_time, self.hyper_param, self.writer)

        # save best model
        best_model_path = './' + self.running_time + "/" + self.hyper_param + "/" + '/logs/global_model/best_model/'
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        if auc > self.pFedMe_global_best_auc:
            self.pFedMe_global_best_auc = auc
            best_model = self.model.state_dict()
            torch.save(best_model, best_model_path + 'best_model.pth')

    # pFedMe
    def evaluate_personalized_model(self, epoch, losses):
        stats = self.test_persionalized_model()
        auc = sum(stats[1]) / len(stats[1])
        loss = sum(stats[2]) / len(stats[2])
        f1 = sum(stats[3]) / len(stats[3])
        acc = sum(stats[4]) / len(stats[4])
        recall = sum(stats[5]) / len(stats[5])
        precision = sum(stats[6]) / len(stats[6])



        print("Personal server\tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(loss, auc, f1, acc, recall, precision))
        server_logging(epoch, loss, auc, f1, acc, recall, precision, self.running_time, self.hyper_param, self.writer)

    # peravg
    def evaluate_one_step(self, epoch):
        for c in self.users:
            c.train_one_step()

        stats = self.global_test()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        auc = sum(stats[1]) / len(stats[1])
        loss = sum(stats[2]) / len(stats[2])
        f1 = sum(stats[3]) / len(stats[3])
        acc = sum(stats[4]) / len(stats[4])
        recall = sum(stats[5]) / len(stats[5])
        precision = sum(stats[6]) / len(stats[6])

        print("Evaluate global model\tloss:{:.4f}\tauc: {:.4f}\tf1: {:.4f}\tacc: {:.4f}\trecall: {:.4f}\tprecision: {:.4f}\t".format(loss, auc, f1, acc, recall, precision))
        global_model_logging(epoch, loss, auc, f1, acc, recall, precision, self.running_time, self.hyper_param, self.writer)







