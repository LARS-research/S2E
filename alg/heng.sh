#!/bin/bash
for((i=16;i<=20;i=i+1))
do
	echo "Lap:" $i
	python heng_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i --result_dir result2/ &
	python heng_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i --result_dir result2/ &
	python heng_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i --result_dir result2/
done
