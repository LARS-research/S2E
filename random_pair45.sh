#!/bin/bash
for((i=1;i<=1;i=i+1))
do
	# echo "Lap:" $i
	python random_main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.45 --result_dir result1 --num_workers 1 --n_iter 17 --test_epoch 32
done
