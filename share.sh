#!/bin/bash
for((i=16;i<=20;i=i+1))
do
	echo "Lap:" $i
	python share_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i --n_epoch 20 &
	python share_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i --n_epoch 20 &
	python share_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --n_iter 20 --n_samples 2 --seed $i --n_epoch 20
done
