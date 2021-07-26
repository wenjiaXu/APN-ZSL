#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./model/main.py \
--dataset CUB \
--calibrated_stacking 0.7 \
--cuda --nepoch 30 --batch_size 64 --train_id 0 --manualSeed 4896 \
--pretrain_epoch 4  --pretrain_lr 1e-4 --classifier_lr 1e-6 \
--xe 1 --attri 1e-2 --regular 5e-6 \
--l_xe 1 --l_attri 1e-1  --l_regular 4e-2 \
--cpt 1e-9 --use_group \
--train_mode 'distributed' --n_batch 300 --ways 8 --shots 3 \


