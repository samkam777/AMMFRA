import os

path = "/huanggx/0002code/multi_FL_all_V1/"

def server_logging(epoch, total_train_loss, total_auc, f1, acc, recall, precision, running_time, hyper_param, writer):

    log_dir_total_train_loss = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total_train_loss/'
    if not os.path.exists(log_dir_total_train_loss):
        os.makedirs(log_dir_total_train_loss)

    with open(log_dir_total_train_loss + 'total_train_loss.txt', 'a+') as f:
        f.write("%s\n" % total_train_loss)

    writer.add_scalar(tag="loss/local_test", scalar_value=total_train_loss, global_step=epoch)

    log_dir_total_auc = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_auc/'
    if not os.path.exists(log_dir_total_auc):
        os.makedirs(log_dir_total_auc)

    with open(log_dir_total_auc + 'total_test_auc.txt', 'a+') as f:
        f.write("%s\n" % total_auc)

    writer.add_scalar(tag="auc/local_test", scalar_value=total_auc, global_step=epoch)

    log_dir_total_f1 = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_f1/'
    if not os.path.exists(log_dir_total_f1):
        os.makedirs(log_dir_total_f1)

    with open(log_dir_total_f1 + 'total_test_f1.txt', 'a+') as f:
        f.write("%s\n" % f1)

    writer.add_scalar(tag="f1/local_test", scalar_value=f1, global_step=epoch)

    log_dir_total_acc = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_acc/'
    if not os.path.exists(log_dir_total_acc):
        os.makedirs(log_dir_total_acc)

    with open(log_dir_total_acc + 'total_test_acc.txt', 'a+') as f:
        f.write("%s\n" % acc)

    writer.add_scalar(tag="acc/local_test", scalar_value=acc, global_step=epoch)

    log_dir_total_recall = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_recall/'
    if not os.path.exists(log_dir_total_recall):
        os.makedirs(log_dir_total_recall)

    with open(log_dir_total_recall + 'total_test_recall.txt', 'a+') as f:
        f.write("%s\n" % recall)

    writer.add_scalar(tag="recall/local_test", scalar_value=recall, global_step=epoch)

    log_dir_total_precision = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total_test_precision/'
    if not os.path.exists(log_dir_total_precision):
        os.makedirs(log_dir_total_precision)

    with open(log_dir_total_precision + 'total_test_precision.txt', 'a+') as f:
        f.write("%s\n" % precision)

    writer.add_scalar(tag="precision/local_test", scalar_value=precision, global_step=epoch)


    log_dir_total = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/server/total/'
    if not os.path.exists(log_dir_total):
        os.makedirs(log_dir_total)

    with open(log_dir_total + 'global_total.txt', 'a+') as f:
        f.write(
            "epoch %d " % epoch + "loss: %.4f " % total_train_loss + "auc: %.4f " % total_auc + "f1: %.4f " % f1 + "acc: %.4f " % acc + "recall: %.4f " % recall + "precision: %.4f \n" % precision)

def global_model_logging(epoch, total_train_loss, total_auc, f1, acc, recall, precision, running_time, hyper_param, writer):

    log_dir_total_train_loss = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total_train_loss/'
    if not os.path.exists(log_dir_total_train_loss):
        os.makedirs(log_dir_total_train_loss)

    with open(log_dir_total_train_loss + 'total_train_loss.txt', 'a+') as f:
        f.write("%s\n" % total_train_loss)

    writer.add_scalar(tag="loss/global_model_test", scalar_value=total_train_loss, global_step=epoch)

    log_dir_total_auc = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total_test_auc/'
    if not os.path.exists(log_dir_total_auc):
        os.makedirs(log_dir_total_auc)

    with open(log_dir_total_auc + 'total_test_auc.txt', 'a+') as f:
        f.write("%s\n" % total_auc)

    writer.add_scalar(tag="auc/global_model_test", scalar_value=total_auc, global_step=epoch)

    log_dir_total_f1 = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total_test_f1/'
    if not os.path.exists(log_dir_total_f1):
        os.makedirs(log_dir_total_f1)

    with open(log_dir_total_f1 + 'total_test_f1.txt', 'a+') as f:
        f.write("%s\n" % f1)

    writer.add_scalar(tag="f1/global_model_test", scalar_value=f1, global_step=epoch)

    log_dir_total_acc = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total_test_acc/'
    if not os.path.exists(log_dir_total_acc):
        os.makedirs(log_dir_total_acc)

    with open(log_dir_total_acc + 'total_test_acc.txt', 'a+') as f:
        f.write("%s\n" % acc)

    writer.add_scalar(tag="acc/global_model_test", scalar_value=acc, global_step=epoch)

    log_dir_total_recall = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total_test_recall/'
    if not os.path.exists(log_dir_total_recall):
        os.makedirs(log_dir_total_recall)

    with open(log_dir_total_recall + 'total_test_recall.txt', 'a+') as f:
        f.write("%s\n" % recall)

    writer.add_scalar(tag="recall/global_model_test", scalar_value=recall, global_step=epoch)

    log_dir_total_precision = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total_test_precision/'
    if not os.path.exists(log_dir_total_precision):
        os.makedirs(log_dir_total_precision)

    with open(log_dir_total_precision + 'total_test_precision.txt', 'a+') as f:
        f.write("%s\n" % precision)

    writer.add_scalar(tag="precision/global_model_test", scalar_value=precision, global_step=epoch)


    log_dir_total = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/global_model/total/'
    if not os.path.exists(log_dir_total):
        os.makedirs(log_dir_total)

    with open(log_dir_total + 'global_total.txt', 'a+') as f:
        f.write(
            "epoch %d " % epoch + "loss: %.4f " % total_train_loss + "auc: %.4f " % total_auc + "f1: %.4f " % f1 + "acc: %.4f " % acc + "recall: %.4f " % recall + "precision: %.4f \n" % precision)

def eps_logging(epoch, epsilons, running_time, hyper_param):

    log_dir_total_eps = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/total_epsilons/'
    if not os.path.exists(log_dir_total_eps):
        os.makedirs(log_dir_total_eps)

    with open(log_dir_total_eps + 'total_epsilons.txt', 'a+') as f:
        f.write("%s\n" % epsilons)

def user_logging(epoch, client_id, total_train_loss, total_auc, f1, acc, recall, precision, running_time, hyper_param, writer):

    log_dir_total_train_loss = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total_train_loss/'
    if not os.path.exists(log_dir_total_train_loss):
        os.makedirs(log_dir_total_train_loss)

    with open(log_dir_total_train_loss + 'total_train_loss.txt', 'a+') as f:
        f.write("%s\n" % total_train_loss)

    writer.add_scalar(tag="loss/user_test", scalar_value=total_train_loss, global_step=epoch)

    log_dir_total_auc = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total_test_auc/'
    if not os.path.exists(log_dir_total_auc):
        os.makedirs(log_dir_total_auc)

    with open(log_dir_total_auc + 'total_test_auc.txt', 'a+') as f:
        f.write("%s\n" % total_auc)

    writer.add_scalar(tag="auc/user_test", scalar_value=total_auc, global_step=epoch)

    log_dir_total_f1 = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/total_test_f1/'
    if not os.path.exists(log_dir_total_f1):
        os.makedirs(log_dir_total_f1)

    with open(log_dir_total_f1 + 'total_test_f1.txt', 'a+') as f:
        f.write("%s\n" % f1)

    writer.add_scalar(tag="f1/user_test", scalar_value=f1, global_step=epoch)

    log_dir_total_acc = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/total_test_acc/'
    if not os.path.exists(log_dir_total_acc):
        os.makedirs(log_dir_total_acc)

    with open(log_dir_total_acc + 'total_test_acc.txt', 'a+') as f:
        f.write("%s\n" % acc)

    writer.add_scalar(tag="acc/user_test", scalar_value=acc, global_step=epoch)

    log_dir_total_recall = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/total_test_recall/'
    if not os.path.exists(log_dir_total_recall):
        os.makedirs(log_dir_total_recall)

    with open(log_dir_total_recall + 'total_test_recall.txt', 'a+') as f:
        f.write("%s\n" % recall)

    writer.add_scalar(tag="recall/user_test", scalar_value=recall, global_step=epoch)

    log_dir_total_precision = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/total_test_precision/'
    if not os.path.exists(log_dir_total_precision):
        os.makedirs(log_dir_total_precision)

    with open(log_dir_total_precision + 'total_test_precision.txt', 'a+') as f:
        f.write("%s\n" % precision)

    writer.add_scalar(tag="precision/user_test", scalar_value=precision, global_step=epoch)

    log_dir_total = path+'/' + running_time + "/" + hyper_param + "/" + '/logs/user/user_' + str(client_id) + '/total/'
    if not os.path.exists(log_dir_total):
        os.makedirs(log_dir_total)

    with open(log_dir_total + 'global_total.txt', 'a+') as f:
        f.write(
            "epoch %d " % epoch + "loss: %.4f " % total_train_loss + "auc: %.4f " % total_auc + "f1: %.4f " % f1 + "acc: %.4f " % acc + "recall: %.4f " % recall + "precision: %.4f \n" % precision)






