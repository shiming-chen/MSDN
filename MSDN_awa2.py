import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.MSDN import MSDN
from core.AWA2DataLoader import AWA2DataLoader
from core.helper_MSDN_AWA2 import eval_zs_gzsl
from global_setting import NFS_path
import importlib
import pdb
import numpy as np

idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataloader = AWA2DataLoader(NFS_path,device)
torch.backends.cudnn.benchmark = True

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr

seed = 87778
# seed = 6379 # for czsl
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

batch_size = 50
nepoches = 50
niters = dataloader.ntrain * nepoches//batch_size
dim_f = 2048
dim_v = 300
init_w2v_att = dataloader.w2v_att
att = dataloader.att#dataloader.normalize_att#
att[att<0] = 0
normalize_att = dataloader.normalize_att
#assert (att.min().item() == 0 and att.max().item() == 1)

trainable_w2v = True
lambda_ = 0.12#0.1 ,0.12 for T-I in GZSL, 0.3 for T-I in CZSL, 0.13 for I-T,0.3 for baseline
bias = 0
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = False

seenclass = dataloader.seenclasses
unseenclass = dataloader.unseenclasses
desired_mass = 1#unseenclass.size(0)/(seenclass.size(0)+unseenclass.size(0))
report_interval = niters//nepoches#10000//batch_size#

model = MSDN(dim_f,dim_v,init_w2v_att,att,normalize_att,
            seenclass,unseenclass,
            lambda_,
            trainable_w2v,normalize_V=True,normalize_F=True,is_conservative=True,
            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
            is_bias=True)
model.to(device)

setup = {'pmp':{'init_lambda':0.1,'final_lambda':0.1,'phase':0.8},
         'desired_mass':{'init_lambda':-1,'final_lambda':-1,'phase':0.8}}
print(setup)

params_to_update = []
params_names = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        params_names.append(name)
        print("\t",name)
#%%
lr = 0.0001
weight_decay = 0.0001#0.000#0.#
momentum = 0.#0.#
#%%
lr_seperator = 1
lr_factor = 1
print('default lr {} {}x lr {}'.format(params_names[:lr_seperator],lr_factor,params_names[lr_seperator:]))
optimizer  = optim.RMSprop( params_to_update ,lr=lr,weight_decay=weight_decay, momentum=momentum)
print('-'*30)
print('learing rate {}'.format(lr))
print('trainable V {}'.format(trainable_w2v))
print('lambda_ {}'.format(lambda_))
print('optimized seen only')
print('optimizer: RMSProp with momentum = {} and weight_decay = {}'.format(momentum,weight_decay))
print('-'*30)

best_performance = [0,0,0]
best_acc = 0
for i in range(0,niters):
    model.train()
    optimizer.zero_grad()
    
    batch_label, batch_feature, batch_att = dataloader.next_batch(batch_size)

    out_package1, out_package2= model(batch_feature)
    
    in_package1 = out_package1
    in_package2 = out_package2
    in_package1['batch_label'] = batch_label
    in_package2['batch_label'] = batch_label
    

    out_package1=model.compute_loss(in_package1)
    out_package2=model.compute_loss(in_package2)
    loss,loss_CE,loss_cal = out_package1['loss']+out_package2['loss'],out_package1['loss_CE']+out_package2['loss_CE'],out_package1['loss_cal']+out_package2['loss_cal']
    constrastive_loss=model.compute_contrastive_loss(in_package1, in_package2)

    loss=loss + 0.0000001*constrastive_loss
    
    loss.backward()
    optimizer.step()
    if i%report_interval==0:
        print('-'*30)
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader,model,device,bias_seen=-bias,bias_unseen=bias)
        
        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H]
        if acc_zs > best_acc:
            best_acc = acc_zs
        print('iter=%d, loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f'%(i,loss.item(),loss_CE.item(),loss_cal.item(),best_performance[0],best_performance[1],best_performance[2],best_acc))
