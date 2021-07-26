import torch
# from gpu_tools import occupy_memory
import time
import os
import argparse
from gpu_tools import occupy_memory
import turtle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='2')  # GPU id
parser.add_argument('--t', type=int, default=20)
global opt
opt = parser.parse_args()



print("_______________\n < rob the GPUs !!! >\n  ---------------\n        \   ^__^\n         \  (oo)\_______\n            (__)\       )\/\ \n                ||----w |\n                ||     ||\n")

if opt.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print('Using gpu:', opt.gpu_id)
    occupy_memory(opt.gpu_id)
    print('Occupy GPU memory in advance.')

print('occupying for {} hours:'.format(opt.t))

for i in range(opt.t):
    print('The {} th hours'.format(i))
    for j in range(100):
        time.sleep(36)
        occupy_memory(opt.gpu_id)
