import torch
from FLAlgorithms.server.serverbase import Server
from FLAlgorithms.users.useravg import UserAVG

class FedAvg(Server):
    def __init__(self, device, args, train_dataset, val_dataset, test_dataset, model, running_time, hyper_param, writer, criterion):
        super().__init__(device, args, train_dataset, val_dataset, test_dataset, model, running_time, hyper_param, writer, criterion)

        for i in range(self.n_users):
            train_data = train_dataset[i]
            val_data = val_dataset[i]
            test_data = test_dataset[i]
            user = UserAVG(device, args, model, train_data, val_data, test_data, i, running_time, hyper_param, criterion)
            self.users.append(user)

    def train(self):

        for global_epoch in range(self.global_epochs):
            print("")
            print("--------------------------global iter: ", global_epoch, " --------------------------")
            self.selected_users = self.select_users(global_epoch, int(self.n_users * self.n_select_users_rate))
            self.send_parameters()
            # print("")
            print("Evaluate average model")
            self.FedAvg_global_evaluate(global_epoch)

            epsilons_list = []
            losses = []
            for user in self.selected_users:
                if self.if_DP:
                    train_loss, epsilons = user.train(global_epoch)  # * user.train_samples
                    epsilons_list.append(epsilons)
                    losses.append(train_loss)
                else:
                    train_loss = user.train(global_epoch)  # * user.train_samples
                    losses.append(train_loss)

            self.aggregate_parameters()

            if self.if_DP:
                eps = sum(epsilons_list) / len(epsilons_list)
                # eps_logging(glob_iter, eps, self.running_time, self.hyper_param)






