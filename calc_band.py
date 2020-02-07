import os
import argparse, sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--test_epoch', type=int, default=20)
parser.add_argument('--eta', type=float, default=3)

args = parser.parse_args()

def main():
    num_models=0
    num_epochs=0
    smax=int(np.floor(np.log(args.test_epoch)/np.log(args.eta)))
    B=(smax+1)*args.test_epoch
    for s in range(smax+1):
        s=smax-s
        n=int(np.ceil((smax+1)/(s+1)*np.power(args.eta,s)))
        r=args.test_epoch/np.power(args.eta,s)
        num_models=num_models+n
        test_runs=n
        for iii in range(s+1):
            ni=np.floor(n/np.power(args.eta,iii))
            ri=np.ceil(r*np.power(args.eta,iii)) # maybe floor?
            num_epochs=num_epochs+test_runs*ri
            test_runs=np.floor(ni/args.eta)
    print(num_models,num_epochs)

if __name__=='__main__':
    main()
