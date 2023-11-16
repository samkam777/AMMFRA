import multiprocessing
import time
import os

running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def fun4(num):
    os.system('python /huanggx/0002code/multi_FL_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.1 --if_DP False --_running_time ' + running_time)
        
def fun5(num):
    os.system('python /huanggx/0002code/multi_FL_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.1 --if_DP True --_running_time ' + running_time)

def fun6(num):
    os.system('python /huanggx/0002code/multi_FL_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.2 --personal_learning_rate 0.005 --if_DP True --_running_time ' + running_time)
    
def fun8(num):
    os.system('python /huanggx/0002code/multi_FL_V2/main.py  --algorithm "pFedMe" --epochs 60 --noise_multiplier 0.05 --if_DP True --_running_time ' + running_time)



if __name__ == '__main__':

    pfed_no_dp = multiprocessing.Process(target=fun4, args=(5,))
    pfed_dp_0_05 = multiprocessing.Process(target=fun8, args=(5,))
    pfed_dp_0_1 = multiprocessing.Process(target=fun5, args=(5,))
    pfed_dp_0_2 = multiprocessing.Process(target=fun6, args=(5,))
    

    pfed_no_dp.start()
    pfed_dp_0_05.start()
    pfed_dp_0_1.start()
    pfed_dp_0_2.start()

    
    
    