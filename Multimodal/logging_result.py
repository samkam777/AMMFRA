import os

def result_logging(epoch, total_train_loss, total_auc, f1, acc, recall, precision, running_time, hyper_param, writer):

    log_dir_total_train_loss = './' + running_time + "/" + hyper_param + "/" + '/logs/total_train_loss/'
    if not os.path.exists(log_dir_total_train_loss):
        os.makedirs(log_dir_total_train_loss)

    with open(log_dir_total_train_loss + 'total_train_loss.txt', 'a+') as f:
        f.write("%s\n" % total_train_loss)

    writer.add_scalar(tag="loss/val", scalar_value=total_train_loss, global_step=epoch)

    log_dir_total_auc = './' + running_time + "/" + hyper_param + "/" + '/logs/total_test_auc/'
    if not os.path.exists(log_dir_total_auc):
        os.makedirs(log_dir_total_auc)

    with open(log_dir_total_auc + 'total_test_auc.txt', 'a+') as f:
        f.write("%s\n" % total_auc)

    writer.add_scalar(tag="auc/val", scalar_value=total_auc, global_step=epoch)

    log_dir_total_f1 = './' + running_time + "/" + hyper_param + "/" + '/logs/total_test_f1/'
    if not os.path.exists(log_dir_total_f1):
        os.makedirs(log_dir_total_f1)

    with open(log_dir_total_f1 + 'total_test_f1.txt', 'a+') as f:
        f.write("%s\n" % f1)

    writer.add_scalar(tag="f1/val", scalar_value=f1, global_step=epoch)

    log_dir_total_acc = './' + running_time + "/" + hyper_param + "/" + '/logs/total_test_acc/'
    if not os.path.exists(log_dir_total_acc):
        os.makedirs(log_dir_total_acc)

    with open(log_dir_total_acc + 'total_test_acc.txt', 'a+') as f:
        f.write("%s\n" % acc)

    writer.add_scalar(tag="acc/val", scalar_value=acc, global_step=epoch)

    log_dir_total_recall = './' + running_time + "/" + hyper_param + "/" + '/logs/total_test_recall/'
    if not os.path.exists(log_dir_total_recall):
        os.makedirs(log_dir_total_recall)

    with open(log_dir_total_recall + 'total_test_recall.txt', 'a+') as f:
        f.write("%s\n" % recall)

    writer.add_scalar(tag="recall/val", scalar_value=recall, global_step=epoch)

    log_dir_total_precision = './' + running_time + "/" + hyper_param + "/" + '/logs/total_test_precision/'
    if not os.path.exists(log_dir_total_precision):
        os.makedirs(log_dir_total_precision)

    with open(log_dir_total_precision + 'total_test_precision.txt', 'a+') as f:
        f.write("%s\n" % precision)

    writer.add_scalar(tag="precision/val", scalar_value=precision, global_step=epoch)


    log_dir_total = './' + running_time + "/" + hyper_param + "/" + '/logs/total/'
    if not os.path.exists(log_dir_total):
        os.makedirs(log_dir_total)

    with open(log_dir_total + 'total.txt', 'a+') as f:
        f.write(
            "epoch %d " % epoch + "loss: %.4f " % total_train_loss + "auc: %.4f " % total_auc + "f1: %.4f " % f1 + "acc: %.4f " % acc + "recall: %.4f " % recall + "precision: %.4f \n" % precision)
