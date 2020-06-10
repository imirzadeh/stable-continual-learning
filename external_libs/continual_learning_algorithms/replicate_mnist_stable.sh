# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
#! /bin/bash

IMP_METHOD='A-GEM'
NUM_RUNS=1
BATCH_SIZE=32
EPS_MEM_BATCH_SIZE=10
MEM_SIZE=1
LOG_DIR='results/mnist'
lr=0.1
ARCH='FC-S'
OPTIM='SGD'
lambda=10



python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1345 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'ER' --log-dir $LOG_DIR
python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1455 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'ER' --log-dir $LOG_DIR
python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1668 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'ER' --log-dir $LOG_DIR


python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1345 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --log-dir $LOG_DIR
python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1455 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --log-dir $LOG_DIR
python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1668 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --log-dir $LOG_DIR


python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1345 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --log-dir $LOG_DIR
python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1455 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --log-dir $LOG_DIR
python3 ./fc_mnist.py --dataset 'rot-mnist' --random-seed 1668 --decay 0.65 --examples-per-task 50000 --arch $ARCH --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --optim $OPTIM --learning-rate $lr --imp-method 'A-GEM' --log-dir $LOG_DIR
