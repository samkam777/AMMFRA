import multiprocessing
import time
import os

running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def fun1(num):
    os.system('python /huanggx/0002code/multi_FL_all_V1/main.py  --algorithm "FedAvg" --model "BST_Multimodal" --epochs 40 --noise_multiplier 0.1 --if_DP False --_running_time ' + running_time)
def fun2(num):
    os.system('python /huanggx/0002code/multi_FL_all_V1/main.py  --algorithm "FedAvg" --model "BST_without" --epochs 40 --noise_multiplier 0.1 --if_DP False --_running_time ' + running_time)
def fun3(num):
    os.system('python /huanggx/0002code/multi_FL_all_V1/main.py  --algorithm "FedAvg" --model "NeuMF" --epochs 40 --noise_multiplier 0.1 --if_DP False --_running_time ' + running_time)
def fun4(num):
    os.system('python /huanggx/0002code/multi_FL_all_V1/main.py  --algorithm "FedAvg" --model "Item2Vec" --epochs 200 --noise_multiplier 0.1 --if_DP False --personal_learning_rate 0.01 --_running_time ' + running_time)
def fun5(num):
    os.system('python /huanggx/0002code/multi_FL_all_V1/main.py  --algorithm "FedAvg" --model "AutoInt" --epochs 40 --noise_multiplier 0.1 --if_DP False --_running_time ' + running_time)
def fun6(num):
    os.system('python /huanggx/0002code/multi_FL_all_V1/main.py  --algorithm "FedAvg" --model "DCNV2" --epochs 40 --noise_multiplier 0.1 --if_DP False --_running_time ' + running_time)






if __name__ == '__main__':
    # "BST_Multimodal", "BST_without", "NeuMF", "Item2Vec", "AutoInt", "DCNV2"

    BST_Multimodal = multiprocessing.Process(target=fun1, args=(5,))
    BST_without = multiprocessing.Process(target=fun2, args=(5,))
    NeuMF = multiprocessing.Process(target=fun3, args=(5,))
    Item2Vec = multiprocessing.Process(target=fun4, args=(5,))
    AutoInt = multiprocessing.Process(target=fun5, args=(5,))
    DCNV2 = multiprocessing.Process(target=fun6, args=(5,))

    
    # BST_Multimodal.start()
    # BST_without.start()
    # NeuMF.start()
    Item2Vec.start()
    # AutoInt.start()
    # DCNV2.start()








    
    