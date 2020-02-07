#!/bin/bash
for((i=6;i<=6;i=i+1))
do
	echo "Lap:" $i
	python heng_100_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 8 --n_samples 5 --seed $i
done
