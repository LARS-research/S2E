#!/bin/bash
for((i=2;i<=5;i=i+1))
# for((i=1;i<=1;i=i+1))
do
	# echo "Lap:" $i
	python bayes_100_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i
done
