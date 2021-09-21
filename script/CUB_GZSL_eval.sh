#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python ./model/main.py \
--dataset CUB \
--calibrated_stacking 0.70 \
--cuda --nepoch 30 --batch_size 64 --train_id 0 --manualSeed 3131 \
--pretrain_epoch 5  --pretrain_lr 1e-4 --classifier_lr 1e-6 \
--xe 1 --attri 1e-2 --regular 5e-5 \
--l_xe 1 --l_attri 1e-1  --l_regular 4e-2 \
--cpt 1e-9 --use_group --gzsl \
--only_evaluate --resume './out/CUB_GZSL_id_0.pth' \



