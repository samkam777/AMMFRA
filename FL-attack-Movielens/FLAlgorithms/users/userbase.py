import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score


class User:
    def __init__(self, device, args, model, train_dataset, val_dataset, test_dataset, numeric_id, running_time, hyper_param):
        self.device = device
        self.model = copy.deepcopy(model)
        self.batch_size = args.batch_size
        self.train_data = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.val_data = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        self.test_data = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=True)
        self.n_train_data_samples = len(self.train_data)*self.batch_size
        print("user_id :{}\tn_train_data_samples :{}".format(numeric_id, self.n_train_data_samples))
        self.id = numeric_id
        self.train_local_epochs = len(self.train_data)
        self.iter_train = iter(self.train_data)
        self.iter_val = iter(self.val_data)
        self.iter_test = iter(self.test_data)
        self.criterion = torch.nn.BCELoss()
        self.personal_lr = args.personal_learning_rate
        self.lamda = args.lamda

        # federated learning personalized model
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.p_local_model = copy.deepcopy(self.model)
        self.persionalized_model_bar = copy.deepcopy(self.model)
        self.localupdate_model = copy.deepcopy(self.model)

        # dp
        self.clip = args.max_grad_norm
        self.noise_multiplier = args.noise_multiplier
        self.if_DP = args.if_DP
        self.delta = args.delta

        self.num_select_users_rate = args.num_select_users_rate


        # attack
        self.local_distance_l = []      # the similarity between local model and global model

        self.pFedMe_local_best_auc = 0



    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def set_poison_parameters(self, model):

        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = -new_param.data.clone()  # -wg
############################################################


###################### user evaluate ######################
    # FedAvg
    def test(self):
        self.model.eval()
        self.update_parameters(self.localupdate_model.parameters())  # 恢复上传前没有加噪声的模型，然后跑验证集
        test_auc, test_losses, test_f1, test_acc, test_recall, test_precision = [], [], [], [], [], []
        for datas in self.val_data:
            datas = [data.to(self.device) for data in datas]
            with torch.no_grad():
                out, target_movie_rating = self.model(datas)
                out = out.flatten()

                # target_movie_rating = 1 if target_movie_rating >= 4 else 0
                target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
                target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

                loss = self.criterion(out, target_movie_rating)
                test_losses.append(loss.item())

                auc, f1, acc, recall, precision = self.compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                                       out.cpu().detach().numpy())
                test_auc.append(auc)
                test_f1.append(f1)
                test_acc.append(acc)
                test_recall.append(recall)
                test_precision.append(precision)
        avg_auc = sum(test_auc) / len(test_auc)
        avg_f1 = sum(test_f1) / len(test_f1)
        avg_acc = sum(test_acc) / len(test_acc)
        avg_recall = sum(test_recall) / len(test_recall)
        avg_precision = sum(test_precision) / len(test_precision)
        avg_losses = sum(test_losses) / len(test_losses)
        # print("test epoch : {}\tloss : {}\tauc : {}".format(epoch, sum(test_losses) / len(test_losses),
        #                                                     sum(test_auc) / len(test_auc)))
        self.update_parameters(self.local_model)  # 恢复上传后添加了加噪声的模型，然后跑验证集

        return avg_auc, avg_losses, avg_f1, avg_acc, avg_recall, avg_precision

    # global evaluation
    def global_test(self):
        self.model.eval()
        test_auc, test_losses, test_f1, test_acc, test_recall, test_precision = [], [], [], [], [], []
        for datas in self.val_data:
            datas = [data.to(self.device) for data in datas]
            # print(len(datas[0]))
            with torch.no_grad():
                out, target_movie_rating = self.model(datas)
                out = out.flatten()

                # target_movie_rating = 1 if target_movie_rating >= 4 else 0
                target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
                target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

                loss = self.criterion(out, target_movie_rating)
                test_losses.append(loss.item())

                auc, f1, acc, recall, precision = self.compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                                       out.cpu().detach().numpy())
                test_auc.append(auc)
                test_f1.append(f1)
                test_acc.append(acc)
                test_recall.append(recall)
                test_precision.append(precision)
        avg_auc = sum(test_auc) / len(test_auc)
        avg_f1 = sum(test_f1) / len(test_f1)
        avg_acc = sum(test_acc) / len(test_acc)
        avg_recall = sum(test_recall) / len(test_recall)
        avg_precision = sum(test_precision) / len(test_precision)
        avg_losses = sum(test_losses) / len(test_losses)

        return avg_auc, avg_losses, avg_f1, avg_acc, avg_recall, avg_precision

    # pFedMe
    def test_persionalized_model(self):
        self.model.eval()
        self.update_parameters(self.persionalized_model_bar.parameters())       # 恢复mata前，然后跑验证集
        test_auc, test_losses, test_f1, test_acc, test_recall, test_precision = [], [], [], [], [], []
        for datas in self.val_data:
            datas = [data.to(self.device) for data in datas]
            with torch.no_grad():
                out, target_movie_rating = self.model(datas)
                out = out.flatten()

                # target_movie_rating = 1 if target_movie_rating >= 4 else 0
                target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
                target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

                loss = self.criterion(out, target_movie_rating)
                test_losses.append(loss.item())

                auc, f1, acc, recall, precision = self.compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                                       out.cpu().detach().numpy())
                test_auc.append(auc)
                test_f1.append(f1)
                test_acc.append(acc)
                test_recall.append(recall)
                test_precision.append(precision)
        avg_auc = sum(test_auc) / len(test_auc)
        avg_f1 = sum(test_f1) / len(test_f1)
        avg_acc = sum(test_acc) / len(test_acc)
        avg_recall = sum(test_recall) / len(test_recall)
        avg_precision = sum(test_precision) / len(test_precision)
        avg_losses = sum(test_losses) / len(test_losses)
        self.update_parameters(self.p_local_model.parameters())                    # meta后，再恢复到更新后的model
        return avg_auc, avg_losses, avg_f1, avg_acc, avg_recall, avg_precision
#############################################################


##################### dateset next batch #####################
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            datas = next(self.iter_train)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_train = iter(self.train_data)
            datas = next(self.iter_train)
        datas = [data.to(self.device) for data in datas]
        return datas

    def get_next_val_batch(self):
        try:
            # Samples a new batch for persionalizing
            datas = next(self.iter_val)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_val = iter(self.val_data)
            datas = next(self.iter_val)
        datas = [data.to(self.device) for data in datas]
        return datas

    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            datas = next(self.iter_test)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_test = iter(self.val_data)
            datas = next(self.iter_test)
        datas = [data.to(self.device) for data in datas]
        return datas
#############################################################


###################### user evaluate ######################
    # pFedMe
    def user_persionalized_evaluate(self, epoch, train_loss):
        auc, losses, f1, acc, recall, precision = self.test_persionalized_model()
        print(
            "user: {}\tloss:{:.4f}\tauc: {:.4f}\tf1:{:.4f}\tacc:{:.4f}\trecall:{:.4f}\tprecision:{:.4f}".format(self.id,
                                                                                                                losses,
                                                                                                                auc, f1,
                                                                                                                acc,
                                                                                                                recall,
                                                                                                                precision))

    # user_logging(epoch, self.id, train_loss, auc, self.running_time, self.hyper_param)

    def user_evaluate(self, epoch, train_loss):
        auc, losses, f1, acc, recall, precision = self.test()
        print(
            "user: {}\tloss:{:.4f}\tauc: {:.4f}\tf1:{:.4f}\tacc:{:.4f}\trecall:{:.4f}\tprecision:{:.4f}".format(self.id,
                                                                                                                losses,
                                                                                                                auc, f1,
                                                                                                                acc,
                                                                                                                recall,
                                                                                                                precision))

    # user_logging(epoch, self.id, train_loss, auc, self.running_time, self.hyper_param)
#############################################################

    def compute_metrics(self, y_true, y_pred):

        auc = roc_auc_score(y_true, y_pred)
        y_pred = (y_pred > 0.5).astype(np.float32)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        return auc, f1, acc, recall, precision


