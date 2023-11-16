from FLAlgorithms.users.userpFed import UserpFed
from FLAlgorithms.server.serverbase import Server


class pFed(Server):
    def __init__(self, device, args, train_dataset, val_dataset, test_dataset, model, running_time, hyper_param, writer):
        super().__init__(device, args, train_dataset, val_dataset, test_dataset, model, running_time, hyper_param, writer)

        for i in range(self.n_users):
            train_data = train_dataset[i]
            val_data = val_dataset[i]
            test_data = test_dataset[i]
            user = UserpFed(device, args, model, train_data, val_data, test_data, i, running_time, hyper_param)
            self.users.append(user)

    def train(self):

        for global_epoch in range(self.global_epochs):
            print("")
            print("--------------------------global iter: ", global_epoch, " --------------------------")
            self.send_parameters()
            print("Evaluate global model")
            # self.pFedMe_global_evaluate(global_epoch)

            epsilons_list = []
            losses = []
            # do update for all users not only selected users
            for user in self.users:
                if self.if_DP:
                    train_loss, epsilons = user.train(global_epoch)  # user.train_samples
                    epsilons_list.append(epsilons)
                    losses.append(train_loss)
                else:
                    train_loss = user.train(global_epoch)  # user.train_samples
                    losses.append(train_loss)

            # choose several users to send back upated model to server
            self.selected_users = self.select_users(global_epoch, int(
                self.n_users * self.n_select_users_rate))

            # Evaluate personalized model on user for each interation
            print("")
            print("Evaluate personalized model")
            self.evaluate_personalized_model(global_epoch, losses)  # 验证的是更新模型前的

            self.median_attention_persionalized_aggregate_parameters()

            if self.if_DP:
                eps = sum(epsilons_list) / len(epsilons_list)
                # eps_logging(glob_iter, eps, self.running_time, self.hyper_param)





