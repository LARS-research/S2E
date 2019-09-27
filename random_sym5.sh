#!/bin/bash
for((i=1;i<=5;i=i+1))
do
	echo "Lap:" $i
	python random_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 5 --n_samples 10 --result_dir result2/ --seed $i
done
