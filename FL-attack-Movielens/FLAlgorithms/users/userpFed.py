from FLAlgorithms.users.userbase import User
import copy
import numpy as np
import torch
import time
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import os

class UserpFed(User):
    def __init__(self, device, args, model, train_dataset, val_dataset, test_dataset, numeric_id, running_time, hyper_param):
        super().__init__(device, args, model, train_dataset, val_dataset, test_dataset, numeric_id, running_time, hyper_param)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.personal_lr)
        self.last_time_local_model = copy.deepcopy(self.model)
        self.save_local_model = copy.deepcopy(self.model)
        self.global_model = copy.deepcopy(self.model)
        self.beta = args.beta

    def custom_lr_lambda(self):
        local_model_param = torch.cat([p.view(-1) for p in self.save_local_model.parameters()]) # 将模型参数展开成一维
        global_model_param = torch.cat([p.view(-1) for p in self.global_model.parameters()])
        local_model_variance = torch.var(local_model_param)
        global_model_variance = torch.var(global_model_param)
        distance = torch.norm(global_model_variance - local_model_variance)
        # print("distance :{}".format(distance))

        learning_rate = self.personal_lr + self.beta*(1/(distance + 1e-4))
        print("learning_rate 1:{}".format(learning_rate))

        return learning_rate

    def train(self, global_epoch):
        self.model.train()

        for param, new_param in zip(self.global_model.parameters(), self.model.parameters()):
            param.data = new_param.data.clone()

        self.optimizer.zero_grad()
        train_losses, train_auc, train_f1, train_acc, train_recall, train_precision = [], [], [], [], [], []

        start_time = time.time()

        self.last_time_local_model = copy.deepcopy(self.model)
        # 增加检测算法



        for datas in self.train_data:
            datas = [data.to(self.device) for data in datas]
            out, target_movie_rating = self.model(datas)
            out = out.flatten()

            # pfed loss
            reg_loss = torch.tensor(0.).to(self.device)
            for p, local_p in zip(self.model.parameters(), self.p_local_model.parameters()):
                reg_loss += (torch.norm(p - local_p, p=2)) ** 2

            target_movie_rating[torch.where(target_movie_rating.lt(4))] = 0  # 小于4置0
            target_movie_rating[torch.where(target_movie_rating.ge(3))] = 1  # 大于3置1

            loss = self.criterion(out, target_movie_rating)
            pfedloss = loss + self.lamda*reg_loss
            pfedloss.backward()

            # clip
            if self.if_DP:
                clip_param_grad = torch.tensor(0.).to(self.device)
                if self.clip is not None:
                    tmp_clip = max(1, self.get_norm() / self.clip)
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad = parameter.grad / tmp_clip
                            clip_param_grad += torch.sum(parameter.grad)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.persionalized_model_bar = copy.deepcopy(self.model)  # clone model to persionalized_model_bar(用于本地预测)

            # loss
            train_losses.append(pfedloss.item())

            auc, f1, acc, recall, precision = self.compute_metrics(target_movie_rating.cpu().detach().numpy(),
                                                                   out.cpu().detach().numpy())
            train_auc.append(auc)
            train_f1.append(f1)
            train_acc.append(acc)
            train_recall.append(recall)
            train_precision.append(precision)

            # self.persionalized_model_bar = copy.deepcopy(self.model)

            for new_param, localweight in zip(self.model.parameters(), self.p_local_model.parameters()):
                # localweight.data = localweight.data - self.learning_rate * (localweight.data - new_param.data)
                localweight.data = new_param.data

        # add noise
        if self.if_DP:
            # self.model = noise_add(self.noise_multiplier, self.model, self.device)
            for p in self.p_local_model.parameters():
                noise = _generate_noise(p, self.noise_multiplier, self.device)
                # noise /= self.train_samples
                p.data += noise

        # update local model as local_weight_upated
        self.update_parameters(self.p_local_model.parameters())


        for param, new_param in zip(self.save_local_model.parameters(), self.p_local_model.parameters()):
            param.data = new_param.data.clone()

        # calculate loss
        train_loss = sum(train_losses) / len(train_losses)

        # evaluate
        # self.user_persionalized_evaluate(global_epoch, train_loss)

        for p in self.optimizer.param_groups:
            p['lr'] = self.custom_lr_lambda()

        epoch_end_time = time.time()

        print(
            "train epoch : {}\tuser : {}\tloss : {:.4f}\tauc : {:.4f}\tf1: {:.4f}\tacc :{:.4f}\trecall :{:.4f}\tprecision :{:.4f}\tspend : {:.4f} h".format(
                global_epoch, self.id,
                sum(train_losses) / len(train_losses), sum(train_auc) / len(train_auc), sum(train_f1) / len(train_f1),
                sum(train_acc) / len(train_acc), sum(train_recall) / len(train_recall),
                sum(train_precision) / len(train_precision), (epoch_end_time - start_time) / 60 / 60))

        # calculate privacy budget

        # save the second model
        save_model_path = './' + self.running_time + "/" + self.hyper_param + "/" + '/logs/user/user_' + str(
            self.id) + '/save_model/'
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        if global_epoch == 2:
            save_model = self.model.state_dict()
            torch.save(save_model, save_model_path + "best_model.pth")

        if self.if_DP:
            epsilons = privacy_budget(self.num_select_users_rate, self.batch_size, global_epoch + 1, self.clip,
                                      self.delta, self.noise_multiplier, self.personal_lr)
            print("training epochs {}  client {}  epsilons:{:.4f}\t".format(global_epoch, self.id, epsilons))
            return train_loss, epsilons
        else:
            return train_loss


    def get_norm(self):
        sum = 0
        for parameter in self.model.parameters():
        # for i in params_a.paramater().keys():
        #     print(parameter.grad)
            if parameter.grad is not None:
                # print("grad is None")
                if len(parameter) == 1:
                    sum += pow(np.linalg.norm(copy.deepcopy(parameter.grad.cpu().numpy()), ord=2), 2)
                else:
                    a = copy.deepcopy(parameter.grad.cpu().numpy())
                    for j in a:
                        x = copy.deepcopy(j.flatten())
                        sum += pow(np.linalg.norm(x, ord=2), 2)
        norm = np.sqrt(sum)
        return norm

    ############## get and set users' parameters ##############
    def set_parameters(self, model):
        similarity = torch.tensor(0.).to(self.device)
        sim_bet_user = torch.tensor(0.).to(self.device)
        for server_param, user_param, last_user_param in zip(model.parameters(), self.get_parameters(), self.last_time_local_model.parameters()):
            # dis += torch.norm(server_param - user_param, p=2)
            similarity += torch.cosine_similarity(server_param.view(-1), user_param.view(-1), dim=0)
            sim_bet_user += torch.cosine_similarity(last_user_param.view(-1), user_param.view(-1), dim=0)
            # user_p += torch.norm(user_param, p=1)
        # print("sim_bet_user: {}\t".format(sim_bet_user))
        # print("similarity: {}\t".format(similarity))

        # # 正常用户在用全局模型更新本地模型前，检测全局模型
        # if len(self.local_distance_l) != 0:  # 不是首次更新
        #
        #
        #
        #     local_dis_l = torch.Tensor(self.local_distance_l)
        #     print("similarity:{}\t".format(local_dis_l))
        #     avg_local_dis = torch.mean(local_dis_l)
        #     print("avg_local_dis:{}\t".format(avg_local_dis))
        #     min_local_dis_l = torch.min(local_dis_l)
        #     max_local_dis_l = torch.max(local_dis_l)
        #
        #     norm_local_dis_l = (similarity - avg_local_dis) / (max_local_dis_l - min_local_dis_l)
        #     norm_local_dis_l = torch.abs(norm_local_dis_l)
        #     norm_local_dis_l = 1 - norm_local_dis_l
        #     print("norm_local_dis_l before normalization: {}\t".format(norm_local_dis_l))
        #     norm_local_dis_l = F.softmax(norm_local_dis_l)
        #     print("norm_local_dis_l after normalization: {}\t".format(norm_local_dis_l))
        #
        # self.local_distance_l.append(similarity)

        if (similarity > 0) and (similarity < 2*sim_bet_user):
            for old_param, new_param, p_local_param, local_param in zip(self.model.parameters(), model.parameters(),
                                                                        self.p_local_model.parameters(), self.local_model):
                old_param.data = new_param.data.clone()
                p_local_param.data = new_param.data.clone()  # personal param update
                local_param.data = new_param.data.clone()  # average param update
        else:
            for old_param, p_local_param, local_param in zip(self.model.parameters(), self.p_local_model.parameters(), self.local_model):
                p_local_param.data = old_param.data.clone()  # personal param update
                local_param.data = old_param.data.clone()  # average param update


        # # 如果本次模型差距大于过往
        # if (sum(self.local_distance_l) / len(self.local_distance_l)) < similarity:
        #     for old_param, new_param, p_local_param, local_param in zip(self.model.parameters(), model.parameters(),self.p_local_model.parameters(),self.local_model):
        #         old_param.data = new_param.data.clone()
        #         p_local_param.data = new_param.data.clone()  # personal param update
        #         local_param.data = new_param.data.clone()  # average param update

def _generate_noise(reference, noise_scale, device):
    return torch.normal(
            0,                              # mean
            noise_scale,                    # std
            reference.data.shape,           # 输出的数据类型, output data size
            device=device,
            # generator=self.random_number_generator,
        )

def privacy_budget(q, datasize, total_epochs, clip_grad, delta, noise_scale, learning_rate):
    delta_s = 2*clip_grad*learning_rate / datasize
    # privacy_budget_ = delta_s * np.sqrt(2*q*total_epochs*np.log(1/delta)) / (noise_scale / datasize)
    privacy_budget_ = delta_s * np.sqrt(2*q*total_epochs*np.log(1/delta)) / noise_scale
    return privacy_budget_






