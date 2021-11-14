import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from core.MSDN import MSDN
from core.SUNDataLoader import SUNDataLoader
from core.helper_MSDN_SUN import eval_zs_gzsl,visualize_attention#,get_attribute_attention_stats
from global_setting import NFS_path
import importlib
import pdb
import numpy as np

idx_GPU = 0
device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
dataloader = SUNDataLoader(NFS_path,device,is_scale=False,is_balance = True)

torch.backends.cudnn.benchmark = True

seed = 214
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print('Randomize seed {}'.format(seed))
#%%
batch_size = 50
nepoches = 80
niters = dataloader.ntrain * nepoches//batch_size
dim_f = 2048
dim_v = 300
init_w2v_att = dataloader.w2v_att
att = dataloader.att#dataloader.normalize_att#
normalize_att = dataloader.normalize_att
#assert (att.min().item() == 0 and att.max().item() == 1)

trainable_w2v = True
lambda_ = 0.001
bias = 0.
prob_prune = 0
uniform_att_1 = False
uniform_att_2 = True

seenclass = dataloader.seenclasses
unseenclass = dataloader.unseenclasses
desired_mass = 1#unseenclass.size(0)/(seenclass.size(0)+unseenclass.size(0))
report_interval = niters//nepoches
#%%
model_gzsl = MSDN(dim_f,dim_v,init_w2v_att,att,normalize_att,
            seenclass,unseenclass,
            lambda_,
            trainable_w2v,normalize_V=False,normalize_F=True,is_conservative=True,
            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
            is_bias=True,non_linear_act=False)
model_gzsl.to(device)
model_gzsl.load_state_dict(torch.load('saved_model/SUN_MSDN_GZSL.pth'))

model_czsl = MSDN(dim_f,dim_v,init_w2v_att,att,normalize_att,
            seenclass,unseenclass,
            lambda_,
            trainable_w2v,normalize_V=False,normalize_F=True,is_conservative=True,
            uniform_att_1=uniform_att_1,uniform_att_2=uniform_att_2,
            prob_prune=prob_prune,desired_mass=desired_mass, is_conv=False,
            is_bias=True)
model_czsl.to(device)
model_czsl.load_state_dict(torch.load('saved_model/SUN_MSDN_CZSL.pth'))




print('-'*30)
acc_seen, acc_novel, H, _ = eval_zs_gzsl(dataloader,model_gzsl,device,bias_seen=-bias,bias_unseen=bias)
_, _, _, acc_zs = eval_zs_gzsl(dataloader,model_czsl,device,bias_seen=-bias,bias_unseen=bias)

print('acc_unseen=%.3f, acc_seen=%.3f, H=%.3f, acc_zs=%.3f'%(acc_novel,acc_seen,H, acc_zs))# %%