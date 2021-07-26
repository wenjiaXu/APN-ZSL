#!/bin/sh
CUDA_VISIBLE_DEVICES=0  python ./model/main.py --calibrated_stacking 0.4 \
--dataset SUN --cuda --nepoch 30 \
--pretrain_epoch 4 --pretrain_lr 1e-3 --classifier_lr 1e-6 --manualSeed 2347 \
--xe 1 --attri 1e-4 --regular 1e-3  \
--l_xe 1 --l_attri 5e-2 --l_regular 5e-3  \
--avg_pool --use_group --cpt 2e-7 --gzsl \
