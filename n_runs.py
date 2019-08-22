import argparse, sys
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--test_epoch', type=int, default=20)
parser.add_argument('--eta', type=float, default=3)

args = parser.parse_args()

total_runs=0
total_epochs=0
print(args.test_epoch,args.eta)
smax=int(np.floor(np.log(args.test_epoch)/np.log(args.eta)))
B=(smax+1)*args.test_epoch
for s in range(smax+1):
    s=smax-s
    n=int(np.ceil((smax+1)/(s+1)*np.power(args.eta,s)))
    r=args.test_epoch/np.power(args.eta,s)
    print(s,n,r)
    T=np.random.rand(n)
    for iii in range(s+1):
        ni=np.floor(n/np.power(args.eta,iii))
        ri=np.ceil(r*np.power(args.eta,iii)) # maybe floor?
        print(ni,ri)
        total_runs+=T.shape[0]
        total_epochs+=T.shape[0]*ri
        idx=np.argsort(T)
        T=T[idx[-int(np.floor(ni/args.eta)):]].copy()

print(total_runs,total_epochs)
