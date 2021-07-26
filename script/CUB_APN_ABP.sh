#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python ./ABP/train_ABP.py --image_embedding APN --dataset CUB --Z_dim 10 --sigma 0.3 --langevin_s 0.1 --langevin_step 5  --manualSeed 4969 --batchsize 64 --nSample 390