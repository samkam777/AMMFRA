import multiprocessing
import time
import os

running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())




def fun5(num):
    os.system(
        'python /huanggx/0002code/multi_FL_attack_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.1 --attack_rate 0.2 --if_DP False --_running_time ' + running_time)

def fun6(num):
    os.system(
        'python /huanggx/0002code/multi_FL_attack_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.1 --attack_rate 0.3 --if_DP False --_running_time ' + running_time)

def fun7(num):
    os.system(
        'python /huanggx/0002code/multi_FL_attack_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.1 --attack_rate 0.4 --if_DP False --_running_time ' + running_time)






if __name__ == '__main__':


    pFedMe_attack_0_2 = multiprocessing.Process(target=fun5, args=(5,))
    pFedMe_attack_0_3 = multiprocessing.Process(target=fun6, args=(5,))
    pFedMe_attack_0_4 = multiprocessing.Process(target=fun7, args=(5,))

    
    

    
    pFedMe_attack_0_2.start()
    pFedMe_attack_0_3.start()
    pFedMe_attack_0_4.start()
    








