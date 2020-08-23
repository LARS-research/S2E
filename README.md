# S2E
ICML'20: Searching to Exploit Memorization Effect in Learning from Corrupted Labels (PyTorch implementation).

=======

This is the code for the paper: [Searching to Exploit Memorization Effect in Learning from Corrupted Labels](https://arxiv.org/abs/1911.02377)
Quanming Yao, Hansi Yang, Bo Han, Gang Niu, James T. Kwok.

## Requirements
Python = 3.7, PyTorch = 1.3.1, NumPy = 1.18.5, SciPy = 1.4.1
All packages can be installed by Conda.

## Running S2E on benchmark dataset with synthetic noise (MNIST, CIFAR-10 and CIFAR-100)
Example usage for MNIST with 50% symmetric noise
```
python heng_mnist_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 10 --n_samples 6
```

CIFAR-10 with 50% symmetric noise
```
python heng_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 10 --n_samples 6
```

And CIFAR-100 with 50% symmetric noise
```
python heng_100_main.py --noise_type symmetric --noise_rate 0.5 --num_workers 1 --n_iter 10 --n_samples 6
```

Or see scripts (.sh files) for a quick start.

## New Opportunities
- Interns, research assistants, and researcher positions are available. See [requirement](http://www.cse.ust.hk/~qyaoaa/pages/job-ad.pdf)
