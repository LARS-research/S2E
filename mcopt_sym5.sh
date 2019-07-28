#!/bin/bash
for((i=1;i<=1;i=i+1))
do
	# echo "Lap:" $i
	python mcopt_main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5 --result_dir result1 --num_workers 1 > mcopt_sym5.txt 2>&1 &
done
