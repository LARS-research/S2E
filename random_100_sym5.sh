#!/bin/bash
for((i=1;i<=1;i=i+1))
do
	# echo "Lap:" $i
	python random_100_main.py --dataset cifar100 --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 3 --n_samples 20
done
