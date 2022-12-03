# MSDN

This is the total codes of paper "**MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning**" accepted to *CVPR'22*. This website includes the following materials for testing and checking our results reported in our paper:

1. The training codes
1. The testing codes
2. The trained model

### Requirements
The code implementation of **MSDN** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.8.8. To install all required dependencies:
```
$ pip install -r requirements.txt
```

## Training

We trained the model on three popular ZSL benchmarks: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html) and [AWA2](http://cvml.ist.ac.at/AwA2/) following the data split of [xlsa17](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip). 
Please follow [TransZero](https://github.com/shiming-chen/TransZero) to prepare datasets and extract visual features.

### Training Script

```
$ python MSDN_cub.py
$ python MSDN_sun.py
$ python MSDN_awa2.py
```
**Note**: Please load the corresponding setting when aiming at the CZSL task.

### Results
We also upload trained models in [test branch](https://github.com/shiming-chen/MSDN). Results of our released models using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings.

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 76.1 | 68.7 | 67.5 | 68.1 |
| SUN | 65.8 | 52.2 | 34.2 | 41.3 |
| AWA2 | 70.1 | 62.0 | 74.5 | 67.7 |

**Note**: All of above results are run on a server with an AMD Ryzen 7 5800X CPU and a NVIDIA RTX A6000 GPU. The training codes will be released soon.

## Testing

### Preparing Dataset and Model

We provide trained models ([Google Drive](https://drive.google.com/drive/folders/1IBGfPXleu4E2BLTI4TlUL1jYSuwahbYC?usp=sharing)) on three different datasets: [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [SUN](http://cs.brown.edu/~gmpatter/sunattributes.html), [AWA2](http://cvml.ist.ac.at/AwA2/) in the CZSL/GZSL setting. You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── saved_model
│   ├── CUB_MSDN_CZSL.pth
│   ├── CUB_MSDN_GZSL.pth
│   ├── SUN_MSDN_CZSL.pth
│   ├── SUN_MSDN_GZSL.pth
│   ├── AWA2_MSDN_CZSL.pth
│   └── AWA2_MSDN_GZSL.pth
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```


## Testing Script
Runing following commands and testing **MSDN** on different dataset:

CUB Dataset: 
```
$ python Test_CUB.py     
```
SUN Dataset:
```
$ python Test_SUN.py     
```
AWA2 Dataset: 
```
$ python Test_AWA2.py     
```

### Results
Results of our released models using various evaluation protocols on three datasets, both in the conventional ZSL (CZSL) and generalized ZSL (GZSL) settings.

| Dataset | Acc(CZSL) | U(GZSL) | S(GZSL) | H(GZSL) |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| CUB | 76.1 | 68.7 | 67.5 | 68.1 |
| SUN | 65.8 | 52.2 | 34.2 | 41.3 |
| AWA2 | 70.1 | 62.0 | 74.5 | 67.7 |

**Note**: All of above results are run on a server with an AMD Ryzen 7 5800X CPU and a NVIDIA RTX A6000 GPU. The training codes will be released soon.

## Citation
If this work is helpful for you, please cite our paper.

```
@InProceedings{Chen2022MSDN,
    author    = {Chen, Shiming and Hong, Ziming and Xie, Guo-Sen and Yang, Wenhan and Peng, Qinmu and Wang, Kai and Zhao, Jian and You, Xinge},
    title     = {MSDN: Mutually Semantic Distillation Network for Zero-Shot Learning},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition ( CVPR )},
    year      = {2022}
}
```


## References
Parts of our codes based on:
* [hbdat/cvpr20_DAZLE](https://github.com/hbdat/cvpr20_DAZLE)
* [shiming-chen/TransZero](https://github.com/shiming-chen/TransZero)
<!--
# Visualization Results
## t-SNE Visualizations
The t-SNE visualization of visual features for seen classes and unseen classes on three datasets, learned by the **"baseline"**, **"MSDN(V->A)"**, **"MSDN(A->V)"**, and **"MSDN(V->A and A->V)"**. The 10 colors denote 10 different seen/unseen classes randomly selected from each dataset.
### CUB Dataset: 
Seen Classes: 
![](images/tsne/cub_tsne_train_seen.png)
Unseen Classes: 
![](images/tsne/cub_tsne_test_unseen.png)

### SUN Dataset:
Seen Classes:  
![](images/tsne/sun_tsne_train_seen.png)
Unseen Class: 
![](images/tsne/sun_tsne_test_unseen.png)

### AWA2 Dataset: 
Seen Classes: 
![](images/tsne/awa2_tsne_train_seen.png)
Unseen Classes: 
![](images/tsne/awa2_tsne_test_unseen.png)

## Attention Maps
Visualization of attention maps for the two mutual attention sub-nets. For each group, the attention maps in the first row are learned by **Attribute->Visual subnet**, the attention maps in the second row  are learned by **Visual->Attribute subnet**. The scores are the attribute scores. 

![](images/t-v/Acadian_Flycatcher_0008_795599.jpg)
![](images/v-t/Acadian_Flycatcher_0008_795599.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/American_Goldfinch_0092_32910.jpg)
![](images/v-t/American_Goldfinch_0092_32910.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/Canada_Warbler_0117_162394.jpg)
![](images/v-t/Canada_Warbler_0117_162394.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/Elegant_Tern_0085_151091.jpg)
![](images/v-t/Elegant_Tern_0085_151091.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/European_Goldfinch_0025_794647.jpg)
![](images/v-t/European_Goldfinch_0025_794647.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/Vesper_Sparrow_0090_125690.jpg)
![](images/v-t/Vesper_Sparrow_0090_125690.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/Western_Gull_0058_53882.jpg)
![](images/v-t/Western_Gull_0058_53882.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/White_Throated_Sparrow_0128_128956.jpg)
![](images/v-t/White_Throated_Sparrow_0128_128956.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/Winter_Wren_0118_189805.jpg)
![](images/v-t/Winter_Wren_0118_189805.jpg)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
![](images/t-v/Yellow_Breasted_Chat_0044_22106.jpg)
![](images/v-t/Yellow_Breasted_Chat_0044_22106.jpg)
-->
