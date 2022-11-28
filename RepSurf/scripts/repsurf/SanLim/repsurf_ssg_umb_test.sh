#!/usr/bin/env bash
set -v

python3 test_cls_sanlim.py \
          --cuda_ops \
          --batch_size 64 \
          --model repsurf.scanobjectnn.repsurf_ssg_umb \
          --epoch 100 \
          --log_dir sanlim2500_100crop \
          --gpus 1 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size 8 \
          --umb_pool sum \
          --num_point 1024
