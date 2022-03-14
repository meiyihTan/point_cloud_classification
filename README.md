# Point Cloud Classification Challenge Task (Shape Recognition)
This repo is a work on Point Cloud Classification with PointNet(https://arxiv.org/abs/1612.00593) as the baseline in pytorch. You can click [here](https://docs.google.com/document/d/1lNPMejT5hPoa-btcG6Bv2YBOYpjswFQNqcnAJXFTXU4/edit#) for the full report and details explanation of this work. 

Problem statement: 

1.To set up a training/inference pipeline for shape recognition of 3D point clouds.
 
Objectives: 

1.To develop a deep learning based model to classify the category(shape recognition of 3D point cloud) of the given input point cloud.

The model is in `pointnet/model.py`.

# Data 
Dataset used is ModelNet40, which is normalizes to [-1,1] (provided by the Akirakan group).

# Training 

```
cd utils
python train_classification.py 
```


# Modifications
6 different modifications/experiments were done in this work.

In `all_experiments.ipynb` notebook, I show the full training and inference pipeline(on model training and inference; the data augmentation and data loading are the same as baseline) for the modification on :

* baseline(U-Net input transform and U-Net feature transform)
* baseline(U-Net input transform and TNet feature transform)
* baseline(TNet input transform and U-Net feature transform)

and also the modification part just on:

* Weighted Cross Entropy Loss(way2)
* Resampling data(Undersampling)
* Data Augmentation(Shuffle)

But the full training and inference pipeline can be find in their respective jupyter notebook.

#The `all_experiments.ipynb` notebook is not in this repo, but located in the ‘/home/meiyih/shape_recognition/ModelNet40_data/’ directory in the server. I grouped all the experiments core cells and have all my explanations in this notebook rather than other ipynb files, but I am unable to download it down. 

#The checkpoint of this work can be found in the `early_stop_ckpt` and `ckpt` folder in the server side.

# Notebooks
* `inputTrans_UNet.ipynb`: a modification which change only input transform T-Net in PointNet to UNet.
* `featureTrans.ipynb`: a modification which change only feature transform T-Net in PointNet to UNet.
* `inputTrans_UNet_featureTrans_UNet.ipynb`: a modification which change both the TNet network of input and feature transform to UNet.
* `weighted_CE_loss1.ipynb`:an implementation of PointNet with weighted CE loss(way 1)
* `weighted_CE_loss2.ipynb`: an implementation of PointNet with weighted CE loss(way 2).
* `undersampling.ipynb`: an implementation of PointNet with undersampling data.
* `baseline_with_shuffle.ipynb`: the implementation of PointNet with shuffled data points.
* `the_baseline.ipynb`:an implementation of PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.

# Performance

## Classification performance


# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
