
import torch
import numpy as np
#%% visualization package
from scipy import ndimage
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
import torch.nn.functional as F
#%%
import pandas as pd
#%%
import pdb
#%%

def mix_up(S1,S2,Y1,Y2): # S: bdwh Y: bk
    device = S1.device
    n = S1.size(0)
    m = torch.empty(n).uniform_().to(device)
    S = torch.einsum('bdwh,b-> bdwh',S1,m)  +   torch.einsum('bdwh,b-> bdwh',S2,1-m)
    Y = torch.einsum('bk,b-> bk',Y1,m)      +   torch.einsum('bk,b-> bk',Y2,1-m)
    return S,Y

def val_gzsl(test_X, test_label, target_classes,in_package,bias = 0):

    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package1, out_package2= model(input)
            
#            if type(output) == tuple:        # if model return multiple output, take the first one
#                output = output[0]
            #output = out_package1['S_pp']
            output = out_package1['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
            predicted_label[start:end] = torch.argmax(output.data, 1)

            start = end

        acc = compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package)
        return acc

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

def val_zs_gzsl(test_X, test_label, unseen_classes,in_package,bias = 0):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        predicted_label_gzsl = torch.LongTensor(test_label.size())
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package1,out_package2 = model(input)
            
#            if type(output) == tuple:        # if model return multiple output, take the first one
#                output = output[0]
#           
            #output = out_package1['S_pp']
            output = out_package1['S_pp']
            
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            
            
            start = end
        acc_gzsl = compute_per_class_acc_gzsl(test_label, predicted_label_gzsl, unseen_classes, in_package)
        acc_zs = compute_per_class_acc_gzsl(test_label, predicted_label_zsl, unseen_classes, in_package)
        acc_zs_t = compute_per_class_acc(map_label(test_label, unseen_classes), predicted_label_zsl_t, unseen_classes.size(0))
        
        # assert np.abs(acc_zs - acc_zs_t) < 0.001
        #print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl,acc_zs_t

def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    
def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):

    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):

        is_class = test_label == target_classes[i]

        per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
#        pdb.set_trace()
    return per_class_accuracies.mean().item()

def eval_zs_gzsl(dataloader,model,device,bias_seen,bias_unseen):
    model.eval()
    # print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = 100
    
    in_package = {'model':model,'device':device, 'batch_size':batch_size}
    
    with torch.no_grad():
        acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen)
        acc_novel,acc_zs = val_zs_gzsl(test_unseen_feature, test_unseen_label, unseenclasses, in_package,bias = bias_unseen)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs
    
def get_heatmap(dataloader,model,device):
    model.eval()
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    eval_size = 100
    n_classes = model.nclass
    n_atts = model.dim_att
    
    heatmap_seen = torch.zeros((n_classes,n_atts))
    heatmap_unseen = torch.zeros((n_classes,n_atts))
    
    with torch.no_grad():
        for c in seenclasses:
            idx_c = torch.squeeze(torch.nonzero(test_seen_label == c))[:eval_size]
            
            batch_c_samples = test_seen_feature[idx_c].to(device)
            out_package = model(batch_c_samples)
            A_p = out_package['A_p']
            heatmap_seen[c] += torch.mean(A_p,dim=0).cpu()
        
        for c in unseenclasses:
            idx_c = torch.squeeze(torch.nonzero(test_unseen_label == c))[:eval_size]
            
            batch_c_samples = test_unseen_feature[idx_c].to(device)
            out_package = model(batch_c_samples)
            A_p = out_package['A_p']
            heatmap_unseen[c] += torch.mean(A_p,dim=0).cpu()
    
    return heatmap_seen.cpu().numpy(),heatmap_unseen.cpu().numpy()

def val_gzsl_k(k,test_X, test_label, target_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        test_label = F.one_hot(test_label, num_classes=n_classes)
        predicted_label = torch.LongTensor(test_label.size()).fill_(0).to(test_label.device)
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package1, out_package2= model(input)
            
#            if type(output) == tuple:        # if model return multiple output, take the first one
#                output = output[0]
#           
            #output = out_package1['S_pp']
            output = out_package1['S_pp']
            output[:,target_classes] = output[:,target_classes]+bias
#            predicted_label[start:end] = torch.argmax(output.data, 1)
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label[start:end] = predicted_label[start:end].scatter_(1,idx_k,1)
            start = end
        
        acc = compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package)
        return acc

def val_zs_gzsl_k(k,test_X, test_label, unseen_classes,in_package,bias = 0,is_detect=False):
    batch_size = in_package['batch_size']
    model = in_package['model']
    device = in_package['device']
    n_classes = in_package["num_class"]
    with torch.no_grad():
        start = 0
        ntest = test_X.size()[0]
        
        test_label_gzsl = F.one_hot(test_label, num_classes=n_classes)
        predicted_label_gzsl = torch.LongTensor(test_label_gzsl.size()).fill_(0).to(test_label.device)
        
        predicted_label_zsl = torch.LongTensor(test_label.size())
        predicted_label_zsl_t = torch.LongTensor(test_label.size())
        for i in range(0, ntest, batch_size):

            end = min(ntest, start+batch_size)

            input = test_X[start:end].to(device)
            
            out_package1,out_package2 = model(input)
            
#            if type(output) == tuple:        # if model return multiple output, take the first one
#                output = output[0]
#           
            #output = out_package1['S_pp']
            output = out_package1['S_pp']
            output_t = output.clone()
            output_t[:,unseen_classes] = output_t[:,unseen_classes]+torch.max(output)+1
            predicted_label_zsl[start:end] = torch.argmax(output_t.data, 1)
            predicted_label_zsl_t[start:end] = torch.argmax(output.data[:,unseen_classes], 1) 
            
            output[:,unseen_classes] = output[:,unseen_classes]+bias
#            predicted_label_gzsl[start:end] = torch.argmax(output.data, 1)
            _,idx_k = torch.topk(output,k,dim=1)
            if is_detect:
                assert k == 1
                detection_mask=in_package["detection_mask"]
                predicted_label_gzsl[start:end] = detection_mask[torch.argmax(output.data, 1)]
            else:
                predicted_label_gzsl[start:end] = predicted_label_gzsl[start:end].scatter_(1,idx_k,1)
            
            start = end
        
        acc_gzsl = compute_per_class_acc_gzsl_k(test_label_gzsl, predicted_label_gzsl, unseen_classes, in_package)
        #print('acc_zs: {} acc_zs_t: {}'.format(acc_zs,acc_zs_t))
        return acc_gzsl,-1

def compute_per_class_acc_k(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
    
def compute_per_class_acc_gzsl_k(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()

    predicted_label = predicted_label.to(device)
    
    hit = test_label*predicted_label
    for i in range(target_classes.size()[0]):

#        is_class = test_label == target_classes[i]
        target = target_classes[i]
        n_pos = torch.sum(hit[:,target])
        n_gt = torch.sum(test_label[:,target])
        per_class_accuracies[i] = torch.div(n_pos.float(),n_gt.float())
        #pdb.set_trace()
    return per_class_accuracies.mean().item()

def eval_zs_gzsl_k(k,dataloader,model,device,bias_seen,bias_unseen,is_detect=False):
    model.eval()
    print('bias_seen {} bias_unseen {}'.format(bias_seen,bias_unseen))
    test_seen_feature = dataloader.data['test_seen']['resnet_features']
    test_seen_label = dataloader.data['test_seen']['labels'].to(device)
    
    test_unseen_feature = dataloader.data['test_unseen']['resnet_features']
    test_unseen_label = dataloader.data['test_unseen']['labels'].to(device)
    
    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses
    
    batch_size = 100
    n_classes = dataloader.ntrain_class+dataloader.ntest_class
    in_package = {'model':model,'device':device, 'batch_size':batch_size,'num_class':n_classes}
    
    if is_detect:
        print("Measure novelty detection k: {}".format(k))
        
        detection_mask = torch.zeros((n_classes,n_classes)).long().to(dataloader.device)
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[seenclasses]=1
        detection_mask[seenclasses,:] = detect_label
        
        detect_label = torch.zeros(n_classes).long().to(dataloader.device)
        detect_label[unseenclasses]=1
        detection_mask[unseenclasses,:]=detect_label
        in_package["detection_mask"]=detection_mask
    
    with torch.no_grad():
        acc_seen = val_gzsl_k(k,test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen,is_detect=is_detect)
        acc_novel,acc_zs = val_zs_gzsl_k(k,test_unseen_feature, test_unseen_label, unseenclasses, in_package,bias = bias_unseen,is_detect=is_detect)

    if (acc_seen+acc_novel)>0:
        H = (2*acc_seen*acc_novel) / (acc_seen+acc_novel)
    else:
        H = 0
        
    return acc_seen, acc_novel, H, acc_zs

def compute_entropy(V):
    eps = 1e-7
    mass = torch.sum(V,dim = 1, keepdim = True) 
    att_n = torch.div(V,mass) 
    e = att_n * torch.log(att_n+eps)  
    e = -1.0 * torch.sum(e,dim=1)  
#    e = torch.mean(e)
    return e

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr

input_size = 224
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])

def visualize_attention(img_ids,alphas_1,alphas_2,S,n_top_attr,attr_name,attr,save_path=None,is_top=True):          #alphas_1: [bir]     alphas_2: [bi]
    n = img_ids.shape[0]
    image_size = 14*16          #one side of the img
    assert alphas_1.shape[1] == alphas_2.shape[1] == len(attr_name)
    r = alphas_1.shape[2]
    h = w =  int(np.sqrt(r))
    for i in range(n):
        fig=plt.figure(i,figsize=(20, 10))
        file_path=img_ids[i]#.decode('utf-8')
        img_name = file_path.split("/")[-1]
#        file_path = img_path+str_id+'.jpg'
        alpha_1 = alphas_1[i]           #[ir]
        alpha_2 = alphas_2[i]           #[i]
        score = S[i]
        # Plot original image
        image = Image.open(file_path)
        if image.mode == 'L':
            image=image.convert('RGB')
        image = data_transforms(image)
        image = image.permute(1,2,0) #[224,244,3] <== [3,224,224] 
        ax = plt.subplot(4, 5, 1)
        plt.imshow(image)
        ax.set_title(img_name,{'fontsize': 10})
#        plt.axis('off')
        
        if is_top:
            idxs_top=np.argsort(-alpha_2)[:n_top_attr]
        else:
            idxs_top=np.argsort(alpha_2)[:n_top_attr]
            
        #pdb.set_trace()
        for idx_ctxt,idx_attr in enumerate(idxs_top):
            ax=plt.subplot(4, 5, idx_ctxt+2)
            plt.imshow(image)
            alp_curr = alpha_1[idx_attr,:].reshape(7,7)
            alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=image_size/h, sigma=10,multichannel=False)
            plt.imshow(alp_img, alpha=0.7)
            ax.set_title("{}\n{}\n{}-{}".format(attr_name[idx_attr],alpha_2[idx_attr],score[idx_attr],attr[idx_attr]),{'fontsize': 10})
#            plt.axis('off')
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path+img_name,dpi=500)
            plt.close()
            
class Logger:
    def __init__(self,filename,cols,is_save=True):
        self.df = pd.DataFrame()
        self.cols = cols
        self.filename=filename
        self.is_save=is_save
    def add(self,values):
        self.df=self.df.append(pd.DataFrame([values],columns=self.cols),ignore_index=True)
    def save(self):
        if self.is_save:
            self.df.to_csv(self.filename)
    def get_max(self,col):
        return np.max(self.df[col])
    
    def is_max(self,col):
        return self.df[col].iloc[-1] >= np.max(self.df[col])
    
def get_attr_entropy(att):  #the lower the more discriminative it is
    eps = 1e-8
    mass=np.sum(att,axis = 0,keepdims=True)
    att_n = np.divide(att,mass+eps)
    entropy = np.sum(-att_n*np.log(att_n+eps),axis=0)
    assert len(entropy.shape)==1
    return entropy