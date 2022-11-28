#!/usr/bin/env bash
set -v

python3 test_cls_sanlim.py \
          --cuda_ops \
          --batch_size 64 \
          --model repsurf.scanobjectnn.repsurf_ssg_umb \
          --epoch 100 \
          --data_dir /data \
          --log_dir sanlim2505_100crop