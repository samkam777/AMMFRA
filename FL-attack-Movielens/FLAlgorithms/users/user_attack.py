import copy
import os
import torch

class UserAttack():
    def __init__(self, device, args, model, numeric_id, running_time, hyper_param):
        # super().__init__(device, args, model, numeric_id, running_time, hyper_param)
        self.total_users = args.n_users
        self.device = device
        self.model = copy.deepcopy(model)
        self.batch_size = args.batch_size
        self.id = numeric_id
        self.n_train_data_samples = 0

        # dp
        self.if_DP = args.if_DP

        # apple
        self.model_c = copy.deepcopy(self.model)
        self.set_apple_poison_parameters(self.model)

        self.model_cs = []
        self.num_clients = args.n_users
        self.ps = [1 / self.num_clients for _ in range(self.num_clients)]
        
        self.running_time = running_time
        self.hyper_param = hyper_param




    def train(self, epochs):

        train_loss = 0
        # save the second model
        save_model_path = './' + self.running_time + "/" + self.hyper_param + "/" + '/logs/user/user_' + str(
            self.id) + '/save_model/'
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        if global_epoch == 1:
            save_model = self.model.state_dict()
            torch.save(save_model, save_model_path + "best_model.pth")
        
        if self.if_DP:
            epsilons = 0
            # print("training epochs {}  client {}  epsilons:{:.4f}\t".format(epochs, self.id, epsilons))
            return train_loss, epsilons
        else:
            return train_loss

    def set_poison_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = (-3)*new_param.data.clone()  # -wg
    def set_apple_poison_parameters(self, model):
        for old_param, new_param in zip(self.model_c.parameters(), model.parameters()):
            old_param.data = (-4)*new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def set_apple_models(self, model_cs):
        self.model_cs = model_cs















