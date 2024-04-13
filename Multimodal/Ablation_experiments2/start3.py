import multiprocessing
import time
import os

running_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def fun4(num):
    os.system('python /huanggx/fullmodal/code/Multimodal/main_wo_txt.py --exp "main_wo_txt" --_running_time ' + running_time)
        
def fun5(num):
    os.system('python /huanggx/fullmodal/code/Multimodal/main_wo_img .py --exp "main_wo_img" --_running_time ' + running_time)

def fun6(num):
    os.system('python /huanggx/fullmodal/code/Multimodal/main_wo_transformer.py --exp "main_wo_transformer" --_running_time ' + running_time)

def fun7(num):
    os.system('python /huanggx/fullmodal/code/Multimodal/main_only_txt.py --exp "main_only_txt" --_running_time ' + running_time)
        
def fun8(num):
    os.system('python /huanggx/fullmodal/code/Multimodal/main_only_img .py --exp "main_only_img" --_running_time ' + running_time)

def fun9(num):
    os.system('python /huanggx/fullmodal/code/Multimodal/main_only_transformer.py --exp "main_only_transformer" --_running_time ' + running_time)

if __name__ == '__main__':

    main_wo_txt = multiprocessing.Process(target=fun4, args=(5,))
    main_wo_img = multiprocessing.Process(target=fun5, args=(5,))
    main_wo_transformer = multiprocessing.Process(target=fun6, args=(5,)) 

    main_only_txt = multiprocessing.Process(target=fun7, args=(5,))
    main_only_img = multiprocessing.Process(target=fun8, args=(5,))
    main_only_transformer = multiprocessing.Process(target=fun9, args=(5,)) 

    main_wo_txt.start()
    main_wo_img.start()
    main_wo_transformer.start()

    main_only_txt.start()
    main_only_img.start()
    main_only_transformer.start()



    
    
    