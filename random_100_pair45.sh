#!/bin/bash
for((i=1;i<=1;i=i+1))
do
	# echo "Lap:" $i
	python random_100_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 5 --n_samples 20 --result_dir result5/ --seed 8
done
