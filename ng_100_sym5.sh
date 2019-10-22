#!/bin/bash
for((i=2;i<=10;i=i+1))
do
	echo "Lap:" $i
	python ng_100_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 20 --n_samples 2 --delta 1 --fisher_samples 9998 --seed $i
done
