# SynFoC
### 1. Introduction

My project is an open-source project based on the paper 'Enhancing Medical Image Segmentation with Collaborative and Contrastive Learning in Mixed-Domain Settings.

### 2. Dataset Construction

The dataset needs to be divided into two folders for training and testing. The training and testing data should be in the format of the "data_format" folder.

### 3. Train

`code/train.py` is the implementation of our method .

Modify the paths in lines 770 to 817 of the code.

```python
if args.dataset == 'fundus':
    train_data_path='../../data/Fundus' # the folder of fundus dataset
elif args.dataset == 'prostate':
    train_data_path="../../data/ProstateSlice" # the folder of prostate dataset
elif args.dataset == 'MNMS':
    train_data_path="../../data/mnms" # the folder of M&Ms dataset
elif args.dataset == 'BUSI':
    train_data_path="../../data/Dataset_BUSI_with_GT" # the folder of BUSI dataset
```

then simply run:

```python
python train.py
```

### 4. Test

To run the evaluation code, please update the path of the dataset in `test.py`:

Modify the paths in lines 248 to 283 of the code.

then simply run:

```
python test.py
```

### 5. DataSets

[Prostate](https://pan.baidu.com/s/1LO2weT01DosfGb3GKLoOvA) with the extraction code: 4no2

[Fundus](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view)

[M&Ms](https://pan.baidu.com/s/1EG1RTIHcJmuzApd8_jvL6w) with the extraction code: cdbs

[BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)

The Prostate and M&Ms datasets have undergone preprocessing in our work, with the original data sourced from [prostate](https://liuquande.github.io/SAML/) and [M&Ms](https://www.ub.edu/mnms/) 

Thanks a lot for their great works.
