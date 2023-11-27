#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# 测试
python src/train.py experiment=h8cloud_rnn logger=csv debug=default
# 训练
# python src/train.py experiment=h8cloud_rnn
