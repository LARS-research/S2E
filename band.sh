#!/bin/bash
for((i=16;i<=20;i=i+1))
do
	echo "Lap:" $i
	python band_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --test_epoch 200 --eta 2.5 --seed $i &
	python band_main.py --noise_type symmetric --noise_rate 0.2 --num_workers 1 --test_epoch 200 --eta 2.5 --seed $i &
	python band_main.py --noise_type pairflip --noise_rate 0.45 --num_workers 1 --test_epoch 200 --eta 2.5 --seed $i
done
