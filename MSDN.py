import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import numpy as np  
import torchvision

  
class MSDN(nn.Module):  
    #####  
    # einstein sum notation  
    # b: Batch size \ f: dim feature=2048 \ v: dim w2v=300 \ r: number of region=196 \ k: number of classes  
    # i: number of attribute=312 
    #####  
    def __init__(self, config, normalize_V = False, normalize_F = False, is_conservative = False,
                 prob_prune=0,uniform_att_1 = False,uniform_att_2 = False, is_conv = False,
                 is_bias = False,bias = 1,non_linear_act=False,
                 loss_type = 'CE',non_linear_emb = False,
                 is_sigmoid = False):  
        super(MSDN, self).__init__()  
        self.config = config
        self.dim_f = config.dim_f  
        self.dim_v = config.dim_v  
        self.nclass = config.num_class
        self.dim_att = config.num_attribute
        self.hidden = self.dim_att//2
        self.non_linear_act = non_linear_act
        self.loss_type = loss_type
        self.w1 = config.w1
        self.w2 = config.w2

        self.att = nn.Parameter(torch.empty(
            self.nclass, self.dim_att), requires_grad=False)
        self.V = nn.Parameter(torch.empty(
            self.dim_att, self.dim_v), requires_grad=True)      
        
        self.W_1 = nn.Parameter(nn.init.normal_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W_2 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W_3 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)

        self.W_1_1 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_f, self.dim_v)), requires_grad=True)
        self.W_2_1 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_v, self.dim_f)), requires_grad=True)
        self.W_3_1 = nn.Parameter(nn.init.zeros_(
            torch.empty(self.dim_f, self.dim_v)), requires_grad=True)

        self.normalize_V = normalize_V  
        self.normalize_F = normalize_F   
        self.is_conservative = is_conservative  
        self.is_conv = is_conv
        self.is_bias = is_bias
        
        if is_bias:
            self.bias = nn.Parameter(torch.tensor(1), requires_grad=False)
            self.mask_bias = nn.Parameter(torch.empty(
                1, self.nclass), requires_grad=False)
        
        self.prob_prune = nn.Parameter(torch.tensor(prob_prune),requires_grad = False) 
         
        self.uniform_att_1 = uniform_att_1
        self.uniform_att_2 = uniform_att_2
        
        self.non_linear_emb = non_linear_emb
        if self.non_linear_emb:
            self.emb_func = torch.nn.Sequential(
                                torch.nn.Linear(self.dim_att, self.dim_att//2),
                                torch.nn.ReLU(),
                                torch.nn.Linear(self.dim_att//2, 1),)
        self.is_sigmoid = is_sigmoid

        # bakcbone
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.resnet101 = nn.Sequential(*list(resnet101.children())[:-2])

      
    def compute_V(self):
        if self.normalize_V:  
            V_n = F.normalize(self.V)
        else:  
            V_n = self.V  
        return V_n
    
    def get_global_feature(self, x):

        N, C, W, H = x.shape
        global_feat = F.avg_pool2d(x, kernel_size=(W, H))
        global_feat = global_feat.view(N, C)

        return global_feat   
        
      
    def forward(self, imgs): 

        Fs = self.resnet101(imgs)
        
        if self.is_conv:
            Fs = self.conv1(Fs)
            Fs = self.conv1_bn(Fs)
            Fs = F.relu(Fs)
        
        shape = Fs.shape

        visualf_ori = self.get_global_feature(Fs)
       

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

        package = {'embed': self.w1 * package1['S_pp']+self.w2 * package2['S_pp']}

        return package
