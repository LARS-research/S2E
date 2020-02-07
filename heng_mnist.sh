#!/bin/bash
# for((i=2;i<=5;i=i+1))
for((i=18;i<=18;i=i+1))
do
	echo "Lap:" $i
	python heng_mnist_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 7 --n_samples 6 --delta 1 --fisher_samples 9998 --seed $i &
	python heng_mnist_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 7 --n_samples 6 --delta 1 --fisher_samples 9998 --seed $i & 
	python heng_mnist_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 7 --n_samples 6 --delta 1 --fisher_samples 9998 --seed $i 
done
