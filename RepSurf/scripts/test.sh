#!/usr/bin/env bash
set -v

python3 test_cls_sanlim.py \
          --cuda_ops \
          --batch_size 64 \
          --epoch 100