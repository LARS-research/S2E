import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions
def loss_softcoteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, num_classes, loss_rate, split_rate):
    temp = F.one_hot(t,num_classes)
    pred_1 = torch.tensor(y_1.tolist(),requires_grad=False)
    pred_1 = pred_1.cuda()
    pred_2 = torch.tensor(y_2.tolist(),requires_grad=False)
    pred_2 = pred_2.cuda()
    t_1 = torch.argmax((1-loss_rate)*temp+loss_rate*pred_2,dim=1)
    t_1 = t_1.cuda()
    t_2 = torch.argmax((1-loss_rate)*temp+loss_rate*pred_1,dim=1)
    t_2 = t_2.cuda()
    # print(t_1,t_2)

    # loss_1 = F.cross_entropy(y_1, t_1, reduce = False)
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = torch.argsort(loss_1).cuda()
    ind_1_cum = ind_1_sorted.cpu() 
    loss_1_sorted = loss_1[ind_1_sorted]

    # loss_2 = F.cross_entropy(y_2, t_2, reduce = False)
    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = torch.argsort(loss_2).cuda()
    ind_2_cum = ind_2_sorted.cpu() 
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - max(forget_rate,split_rate)
    throw_rate = 1 - min(forget_rate,split_rate)
    # remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    num_throw = int(throw_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_cum[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_cum[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    ind_1_midate=ind_1_sorted[num_remember:num_throw]
    ind_2_midate=ind_2_sorted[num_remember:num_throw]
    ind_1_downdate=ind_1_sorted[num_throw:]
    ind_2_downdate=ind_2_sorted[num_throw:]
    # exchange
    if len(ind_1_midate)==0:
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update]) 
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    else:
        loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update]) + F.cross_entropy(y_1[ind_2_midate], t_2[ind_2_midate])
        loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update]) + F.cross_entropy(y_2[ind_1_midate], t_1[ind_1_midate])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

def loss_curve(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    batch_size=len(loss_1_sorted)
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * batch_size)

    ind1=np.arange(128,dtype=np.int)
    ind1=ind1[noise_or_not[ind]]
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2, (torch.sum(loss_1[ind1])+torch.sum(loss_2[ind1]))/2, (torch.sum(loss_1)+torch.sum(loss_2)-torch.sum(loss_1[ind1])-torch.sum(loss_2[ind1]))/2 

def loss_selfteaching(y_1, y_2, t, forget_rate, ind, noise_or_not, num_classes):
    temp = F.one_hot(t,num_classes)
    pred_1 = torch.tensor(y_1.tolist(),requires_grad=False)
    pred_1 = pred_1.cuda()
    pred_2 = torch.tensor(y_2.tolist(),requires_grad=False)
    pred_2 = pred_2.cuda()
    t_1 = torch.argmax((1-forget_rate)*temp+forget_rate*pred_2,dim=1)
    t_1 = t_1.cuda()
    t_2 = torch.argmax((1-forget_rate)*temp+forget_rate*pred_1,dim=1)
    t_2 = t_2.cuda()
    # print(t_1,t_2)

    loss_1 = F.cross_entropy(y_1, t_1, reduce = False)
    # ind_1_sorted = torch.argsort(loss_1).cuda()
    # ind_1_cum = ind_1_sorted.cpu() 
    # ind_1_cum = ind_1_sorted 
    # loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t_2, reduce = False)
    # ind_2_sorted = torch.argsort(loss_2).cuda()
    # ind_2_cum = ind_2_sorted.cpu() 
    # ind_2_cum = ind_2_sorted 
    # loss_2_sorted = loss_2[ind_2_sorted]

    # remember_rate = 1 - forget_rate
    num_remember = len(loss_1)

    pure_ratio_1 = 1
    pure_ratio_2 = 1

    # ind_1_update=ind_1_sorted[:num_remember]
    # ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = loss_1.cuda()
    loss_2_update = loss_2.cuda()
    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = torch.argsort(loss_1).cuda()
    ind_1_cum = ind_1_sorted.cpu() 
    # ind_1_cum = ind_1_sorted 
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = torch.argsort(loss_2).cuda()
    ind_2_cum = ind_2_sorted.cpu() 
    # ind_2_cum = ind_2_sorted 
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_cum[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_cum[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

def loss_curriculum(y_1, y_2, t, forget_rate, ind):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    # loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    # loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    loss_1_update = F.cross_entropy(y_1[:num_remember], t[:num_remember])
    loss_2_update = F.cross_entropy(y_2[:num_remember], t[:num_remember])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, 1, 1

def loss_3teaching(y_1, y_2, y_3, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    loss_3 = F.cross_entropy(y_3, t, reduce = False)
    ind_3_sorted = np.argsort(loss_3.data).cuda()
    loss_3_sorted = loss_3[ind_3_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_3 = np.sum(noise_or_not[ind[ind_3_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=np.intersect1d(ind_2_sorted[:num_remember],ind_3_sorted[:num_remember],assume_unique=True)
    ind_2_update=np.intersect1d(ind_3_sorted[:num_remember],ind_1_sorted[:num_remember],assume_unique=True)
    ind_3_update=np.intersect1d(ind_1_sorted[:num_remember],ind_2_sorted[:num_remember],assume_unique=True)
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    loss_2_update = F.cross_entropy(y_2[ind_2_update], t[ind_2_update])
    loss_3_update = F.cross_entropy(y_3[ind_3_update], t[ind_3_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, torch.sum(loss_3_update)/num_remember, pure_ratio_1, pure_ratio_2, pure_ratio_3

