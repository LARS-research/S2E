#!/bin/bash
for((i=17;i<=20;i=i+1))
# for((i=1;i<=1;i=i+1))
do
	echo "Lap:" $i
	python grad_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 20 --n_samples 2 --delta 1 --fisher_samples 9998 --seed $i &
	python grad_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 20 --n_samples 2 --delta 1 --fisher_samples 9998 --seed $i &
	python grad_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 20 --n_samples 2 --delta 1 --fisher_samples 9998 --seed $i 
done
