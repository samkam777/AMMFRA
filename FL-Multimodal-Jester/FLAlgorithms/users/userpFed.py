from FLAlgorithms.users.userbase import User
import copy
import numpy as np
import torch
import time
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR

class UserpFed(User):
    def __init__(self, device, args, model, train_dataset, val_dataset, test_dataset, numeric_id, running_time, hyper_param):
        super().__init__(device, args, model, train_dataset, val_dataset, test_dataset, numeric_id, running_time, hyper_param)

        self.personal_lr = 0.0005
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.personal_lr)
        self.save_local_model = copy.deepcopy(self.model)
        self.global_model = copy.deepcopy(self.model)
        self.beta = args.beta
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.custom_lr_lambda())
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.personal_lr*(epoch+1))
        self.local_epochs = len(self.train_data)

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
        # self.custom_lr_lambda(optimizer=self.optimizer, epoch=global_epoch)

        for param, new_param in zip(self.global_model.parameters(), self.model.parameters()):
            param.data = new_param.data.clone()

        self.optimizer.zero_grad()
        train_losses, train_auc, train_f1, train_acc, train_recall, train_precision = [], [], [], [], [], []
        start_time = time.time()
        # for datas in self.train_data:
        for epoch in range(1, self.local_epochs + 1):  # local update
            datas = self.get_next_train_batch()
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

            auc, f1, acc, recall, precision = self.compute_metrics(target_movie_rating.cpu().detach().numpy(), out.cpu().detach().numpy())
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

        # self.scheduler.step()
        # print("learning rate2 :{}".format(self.scheduler.get_last_lr()))
        for p in self.optimizer.param_groups:
            p['lr'] = self.custom_lr_lambda()

        epoch_end_time = time.time()

        print("train epoch : {}\tuser : {}\tloss : {:.4f}\tauc : {:.4f}\tf1: {:.4f}\tacc :{:.4f}\trecall :{:.4f}\tprecision :{:.4f}\tspend : {:.4f} h".format(global_epoch, self.id,
            sum(train_losses) / len(train_losses), sum(train_auc) / len(train_auc), sum(train_f1) / len(train_f1), sum(train_acc) / len(train_acc), sum(train_recall) / len(train_recall), sum(train_precision) / len(train_precision), (epoch_end_time - start_time) / 60 / 60))

        # calculate privacy budget
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






