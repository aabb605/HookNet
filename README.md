# An Efficient Fabric Defect Detection Method based on EfficientDet.
## Contents
- <a href='#Data Preparation'>Data Preparation</a>
- <a href='#Installation'>Installation</a>
- <a href='#Train'>Train</a>
- <a href='#Evaluation'>Evaluation</a>
- <a href='#Test'>Test</a>
- <a href='#Reference'>Reference</a>

## Data Preparation
### TILDA dataset 
Download from [https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html](https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html)
### DAGM2007 dataset
Download from [https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection](https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection)

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- The trained weights can be found [here](https://github.com/bubbliiiing/efficientdet-pytorch/releases)
- Annotate defective data set data and Generate VOC format (VOCdevkit/VOC2007/JPEGImages and VOCdevkit/VOC2007/Annotations)
  * Note: Install [labelImg](https://github.com/tzutalin/labelImg) to label data
  
## Train
- Edit the classes to fit your dataset in model_data/new_classes.txt
- Edit the path of the weight and the .txt in efficientdet.py | train.py
```python
"model_path"   : 'model_data/efficientdet-d0.pth ',
"classes_path" : 'model_data/new_classes.txt'
```
- Modify the "phi" value
- Run voc_annotation.py to generate the corresponding .txt file before training  
- Divide training set and test set (VOCdevkit/VOC2007/ImageSets/Main/test.txt || trainval.txt || train.txt || val.txt)
  * Note: The ratio of training set to test set can be from 6:4 to 8:2
- Run train.py 

## Evaluation
- Edit the file path of trained weight model in efficientdet.py
- Run get_map.py
- Run summary.py

## Test
See file "test results" of my experiments for detail
- Run predict.py

## Reference
[https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
[https://github.com/bubbliiiing/efficientdet-pytorch](https://github.com/bubbliiiing/efficientdet-pytorch)
