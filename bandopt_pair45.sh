#!/bin/bash
for((i=1;i<=1;i=i+1))
do
	# echo "Lap:" $i
	python bandopt_main.py --dataset cifar10 --noise_type pairflip --noise_rate 0.45 --result_dir result1 --num_workers 1 --test_epoch 32
done
