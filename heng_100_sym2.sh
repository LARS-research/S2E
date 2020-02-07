#!/bin/bash
for((i=4;i<=4;i=i+1))
do
	echo "Lap:" $i
	python heng_100_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 2 --n_samples 20 --seed $i
done
