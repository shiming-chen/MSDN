# -*- coding: utf-8 -*-  
"""  
Created on Thu Jul  4 17:39:45 2019  
  
@author: badat  
"""  
import tensorflow as tf
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
#%%  
  
class MSDN(nn.Module):  
    #####  
    # einstein sum notation  
    # b: Batch size \ f: dim feature=2048 \ v: dim w2v=300 \ r: number of region=196 \ k: number of classes  
    # i: number of attribute=312 
    #####  
    def __init__(self,dim_f,dim_v,  
                 init_w2v_att,att,normalize_att,  
                 seenclass,unseenclass, 
                 lambda_,  
                 trainable_w2v = False, normalize_V = False, normalize_F = False, is_conservative = False,
                 prob_prune=0,desired_mass = -1,uniform_att_1 = False,uniform_att_2 = False, is_conv = False,
                 is_bias = False,bias = 1,non_linear_act=False,
                 loss_type = 'CE',non_linear_emb = False,
                 is_sigmoid = False):  
        super(MSDN, self).__init__()  
        self.dim_f = dim_f  
        self.dim_v = dim_v  
        self.dim_att = att.shape[1]  
        self.nclass = att.shape[0]  
        self.hidden = self.dim_att//2
        self.init_w2v_att = init_w2v_att
        self.non_linear_act = non_linear_act
        self.loss_type = loss_type
        if is_conv:
            r_dim = dim_f//2
            self.conv1 = nn.Conv2d(dim_f, r_dim, 2) #[2x2] kernel with same input and output dims
            print('***Reduce dim {} -> {}***'.format(self.dim_f,r_dim))
            self.dim_f = r_dim
            self.conv1_bn = nn.BatchNorm2d(self.dim_f)
            
            
        if init_w2v_att is None:  
            self.V = nn.Parameter(nn.init.normal_(torch.empty(self.dim_att,self.dim_v)),requires_grad = True)  
        else:
            self.init_w2v_att = F.normalize(torch.tensor(init_w2v_att))
            self.V = nn.Parameter(self.init_w2v_att.clone(),requires_grad = trainable_w2v)  
        
        self.att = nn.Parameter(F.normalize(torch.tensor(att)),requires_grad = False)          
        
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#  
        self.W_2 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) #nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#  
        ## second layer attenion conditioned on image features
        self.W_3 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True)
        
        self.W_1_1 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_f,self.dim_v)),requires_grad = True)#nn.utils.weight_norm(nn.Linear(self.dim_v,self.dim_f,bias=False))#  
        self.W_2_1 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_v,self.dim_f)),requires_grad = True) 
        self.W_3_1 = nn.Parameter(nn.init.zeros_(torch.empty(self.dim_f,self.dim_v)),requires_grad = True)
        
        ## Compute the similarity between classes  
        self.P = torch.mm(self.att,torch.transpose(self.att,1,0))  
        assert self.P.size(1)==self.P.size(0) and self.P.size(0)==self.nclass  
        self.weight_ce = nn.Parameter(torch.eye(self.nclass).float(),requires_grad = False)#nn.Parameter(torch.tensor(weight_ce).float(),requires_grad = False)  

        self.normalize_V = normalize_V  
        self.normalize_F = normalize_F   
        self.is_conservative = is_conservative  
        self.is_conv = is_conv
        self.is_bias = is_bias
        
        self.seenclass = seenclass  
        self.unseenclass = unseenclass  
        self.normalize_att = normalize_att   
        
        if is_bias:
            self.bias = nn.Parameter(torch.tensor(bias),requires_grad = False)
            mask_bias = np.ones((1,self.nclass))
            mask_bias[:,self.seenclass.cpu().numpy()] *= -1
            self.mask_bias = nn.Parameter(torch.tensor(mask_bias).float(),requires_grad = False)
        
        if desired_mass == -1:  
            self.desired_mass = self.unseenclass.size(0)/self.nclass#nn.Parameter(torch.tensor(self.unseenclass.size(0)/self.nclass),requires_grad = False)#nn.Parameter(torch.tensor(0.1),requires_grad = False)#  
        else:  
            self.desired_mass = desired_mass#nn.Parameter(torch.tensor(desired_mass),requires_grad = False)#nn.Parameter(torch.tensor(self.unseenclass.size(0)/self.nclass),requires_grad = False)#  
        self.prob_prune = nn.Parameter(torch.tensor(prob_prune),requires_grad = False) 
         
        self.lambda_ = lambda_
        self.loss_att_func = nn.BCEWithLogitsLoss()
        self.log_softmax_func = nn.LogSoftmax(dim=1)  
        self.uniform_att_1 = uniform_att_1
        self.uniform_att_2 = uniform_att_2
        
        self.non_linear_emb = non_linear_emb
        
        
        print('-'*30)  
        print('Configuration')  
        
        print('loss_type {}'.format(loss_type))
        
        if self.is_conv:
            print('Learn CONV layer correct')
        
        if self.normalize_V:  
            print('normalize V')  
        else:  
            print('no constraint V')  
              
        if self.normalize_F:  
            print('normalize F')  
        else:  
            print('no constraint F')  
              
        if self.is_conservative:  
            print('training to exclude unseen class [seen upperbound]')  
        if init_w2v_att is None:  
            print('Learning word2vec from scratch with dim {}'.format(self.V.size()))  
        else:  
            print('Init word2vec')  
        
        if self.non_linear_act:
            print('Non-linear relu model')
        else:
            print('Linear model')
        
        print('loss_att {}'.format(self.loss_att_func))  
        print('Bilinear attention module')  
        print('*'*30)  
        print('Measure w2v deviation')
        if self.uniform_att_1:
            print('WARNING: UNIFORM ATTENTION LEVEL 1')
        if self.uniform_att_2:
            print('WARNING: UNIFORM ATTENTION LEVEL 2')
        print('Compute Pruning loss {}'.format(self.prob_prune))  
        if self.is_bias:
            print('Add one smoothing')
        print('Second layer attenion conditioned on image features')
        print('-'*30)  
        
        if self.non_linear_emb:
            print('non_linear embedding')
            self.emb_func = torch.nn.Sequential(
                                torch.nn.Linear(self.dim_att, self.dim_att//2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(self.dim_att//2, 1),
                            )
        
        self.is_sigmoid = is_sigmoid
        if self.is_sigmoid:
            print("Sigmoid on attr score!!!")
        else:
            print("No sigmoid on attr score")

    
    def compute_loss_rank(self,in_package):  
        # this is pairwise ranking loss  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp']  
        
        batch_label_idx = torch.argmax(batch_label,dim = 1)
        
        s_c = torch.gather(S_pp,1,batch_label_idx.view(-1,1))  
        if self.is_conservative:  
            S_seen = S_pp  
        else:  
            S_seen = S_pp[:,self.seenclass]  
            assert S_seen.size(1) == len(self.seenclass)  
          
        margin = 1-(s_c-S_seen)  
        loss_rank = torch.max(margin,torch.zeros_like(margin))  
        loss_rank = torch.mean(loss_rank)  
        return loss_rank  
      
    def compute_loss_Self_Calibrate(self,in_package):  
        S_pp = in_package['S_pp']  
        Prob_all = F.softmax(S_pp,dim=-1)
        Prob_unseen = Prob_all[:,self.unseenclass]  
        assert Prob_unseen.size(1) == len(self.unseenclass)  
        mass_unseen = torch.sum(Prob_unseen,dim=1)  
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp  
      
    def compute_V(self):
        if self.normalize_V:  
            V_n = F.normalize(self.V)
        else:  
            V_n = self.V  
        return V_n
    
    def compute_aug_cross_entropy(self,in_package):  
        batch_label = in_package['batch_label']  
        S_pp = in_package['S_pp']  
        
        Labels = batch_label
            
        if self.is_bias:
            S_pp = S_pp - self.vec_bias         # remove the margin +1/-1 from prediction scores
        
        if not self.is_conservative:  
            S_pp = S_pp[:,self.seenclass]  
            Labels = Labels[:,self.seenclass]  
            assert S_pp.size(1) == len(self.seenclass)  
        
        Prob = self.log_softmax_func(S_pp)  
          
        loss = -torch.einsum('bk,bk->b',Prob,Labels)  
        loss = torch.mean(loss)  
        return loss  
    
    def compute_loss(self,in_package):
        
        if len(in_package['batch_label'].size()) == 1:
            in_package['batch_label'] = self.weight_ce[in_package['batch_label']]  
        
        ## loss rank  
        if self.loss_type == 'CE':
            loss_CE = self.compute_aug_cross_entropy(in_package)
        elif self.loss_type == 'rank':
            loss_CE = self.compute_loss_rank(in_package)  
        else:
            raise Exception('Unknown loss type')
        
        ## loss self-calibration  
        loss_cal = self.compute_loss_Self_Calibrate(in_package)
        
        ## total loss  
        loss = loss_CE #+ self.lambda_*loss_cal
          
        out_package = {'loss':loss,'loss_CE':loss_CE,
                       'loss_cal':loss_cal}  
          
        return out_package  
    
    def compute_contrastive_loss(self, in_package1, in_package2):
        S_pp1,S_pp2=in_package1['S_pp'], in_package2['S_pp']
        wt = (S_pp1-S_pp2).pow(2)
        wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
        loss = wt * (S_pp1-S_pp2).abs()
        loss= (loss.sum()/loss.size(0))

        #JSD
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        p_output = F.softmax(S_pp1)
        q_output = F.softmax(S_pp2)
        log_mean_output = ((p_output + q_output )/2).log()
        loss+=(KLDivLoss(log_mean_output, q_output) + KLDivLoss(log_mean_output, p_output))/2

        return loss


        #KL-Divergence
        # logp_x = F.log_softmax(S_pp2, dim=-1)
        # p_y = F.softmax(S_pp1, dim=-1)
        # kl_mean = F.kl_div(logp_x, p_y, reduction='mean')
        
        
    
    def get_global_feature(self, x):

        N, C, W, H = x.shape
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)
        # gs_feat = torch.einsum('bf,fi->bi', global_feat, self.V_w)

        # gs_feat_norm = torch.norm(gs_feat, p=2, dim = 1).unsqueeze(1).expand_as(gs_feat)
        # gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)

        return global_feat   
        
      
    def forward(self,Fs): 
        
        if self.is_conv:
            Fs = self.conv1(Fs)
            Fs = self.conv1_bn(Fs)
            Fs = F.relu(Fs)
        
        shape = Fs.shape

        visualf_ori = self.get_global_feature(Fs)
       
       ##########################base-model################################
        # global_feature = self.get_global_feature(Fs)
        # temp_norm = torch.norm(self.att, p=2, dim=1).unsqueeze(1).expand_as(self.att)
        # seen_att_normalized = self.att.div(temp_norm + 1e-5)

        # S_pp = torch.einsum('bi,ki->bk', global_feature, seen_att_normalized)
        # package0 = {'S_pp':S_pp}  

        # return package0
        ##########################base-model################################
        
        Fs = Fs.reshape(shape[0],shape[1],shape[2]*shape[3]) # batch x 2048 x 49
        
        R = Fs.size(2)  # 49
        B = Fs.size(0)  # batch
        V_n = self.compute_V() # 312x300
        
        if self.normalize_F and not self.is_conv:  # true
            Fs = F.normalize(Fs,dim = 1)  
        
        
        ##########################Text-Image################################
        
        ## Compute attribute score on each image region
        S = torch.einsum('iv,vf,bfr->bir',V_n,self.W_1,Fs) # batchx312x49
        
        if self.is_sigmoid:
            S=torch.sigmoid(S)
        
        ## Ablation setting
        A_b = Fs.new_full((B,self.dim_att,R),1/R)
        A_b_p = self.att.new_full((B,self.dim_att),fill_value = 1)
        S_b_p = torch.einsum('bir,bir->bi',A_b,S)  
        S_b_pp = torch.einsum('ki,bi,bi->bk',self.att,A_b_p,S_b_p)
        ##
        
        ## compute Dense Attention
        A = torch.einsum('iv,vf,bfr->bir',V_n,self.W_2,Fs)  
        A = F.softmax(A,dim = -1)                  
        
        F_p = torch.einsum('bir,bfr->bif',A,Fs)    
        if self.uniform_att_1: # false
            S_p = torch.einsum('bir,bir->bi',A_b,S)    
        else:
            S_p = torch.einsum('bir,bir->bi',A,S)       
        
        if self.non_linear_act: # false
            S_p = F.relu(S_p)
        ## 
        
        ## compute Attention over Attribute
        A_p = torch.einsum('iv,vf,bif->bi',V_n,self.W_3,F_p) #eq. 6
        A_p = torch.sigmoid(A_p) 
        ##  
        
        if self.uniform_att_2:  # true
            S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_b_p,S_p)    
        else:
            # S_pp = torch.einsum('ki,bi,bi->bik',self.att,A_p,S_p)     
            S_pp = torch.einsum('ki,bi->bik',self.att,S_p)
            
        S_attr = torch.einsum('bi,bi->bi',A_b_p,S_p)
            
        if self.non_linear_emb:
            S_pp = torch.transpose(S_pp,2,1)    #[bki] <== [bik]
            S_pp = self.emb_func(S_pp)          #[bk1] <== [bki]
            S_pp = S_pp[:,:,0]                  #[bk] <== [bk1]
        else:
            S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik]
        
        # augment prediction scores by adding a margin of 1 to unseen classes and -1 to seen classes
        if self.is_bias:
            self.vec_bias = self.mask_bias*self.bias
            S_pp = S_pp + self.vec_bias
                
        ## spatial attention supervision   
        Pred_att = torch.einsum('iv,vf,bif->bi',V_n,self.W_1,F_p) 
        package1 = {'S_pp':S_pp,'Pred_att':Pred_att,'S_p':S_p,'S_b_pp':S_b_pp,'A_p':A_p,'A':A,'S_attr':S_attr,'visualf_ori':visualf_ori,'visualf_a_v':F_p}
        
        ##########################Image-Text################################
        
        ## Compute attribute score on each image region

        S = torch.einsum('bfr,fv,iv->bri',Fs,self.W_1_1,V_n) # batchx49x312
        # S = torch.einsum('iv,vf,bfr->bir',V_n,self.W_1_1,Fs)
        if self.is_sigmoid:
            S=torch.sigmoid(S)
        

        
        ## compute Dense Attention
        A = torch.einsum('iv,vf,bfr->bir',V_n,self.W_2_1,Fs)
        A = F.softmax(A,dim = 1)                 
        
        v_a = torch.einsum('bir,iv->brv',A,V_n)    

        S_p = torch.einsum('bir,bri->bi',A,S)       
        
        if self.non_linear_act: # false
            S_p = F.relu(S_p)

        

        S_pp = torch.einsum('ki,bi->bik',self.att,S_p)     
            
        S_attr = 0#torch.einsum('bi,bi->bi',A_b_p,S_p)
            
        if self.non_linear_emb:
            S_pp = torch.transpose(S_pp,2,1)    #[bki] <== [bik]
            S_pp = self.emb_func(S_pp)          #[bk1] <== [bki]
            S_pp = S_pp[:,:,0]                  #[bk] <== [bk1]
        else:
            S_pp = torch.sum(S_pp,axis=1)        #[bk] <== [bik]
        
        # augment prediction scores by adding a margin of 1 to unseen classes and -1 to seen classes
        if self.is_bias:
            self.vec_bias = self.mask_bias*self.bias
            S_pp = S_pp + self.vec_bias
                
        ## spatial attention supervision   
        package2 = {'S_pp':S_pp,'visualf_v_a':v_a, 'S_p':S_p, 'A':A}

        return package1, package2

# %%
#
