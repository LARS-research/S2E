#!/bin/bash
for((i=11;i<=11;i=i+1))
do
	echo "Lap:" $i
	python co_100_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i
done
