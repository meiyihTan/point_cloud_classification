# PointNet.pytorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The model is in `pointnet/model.py`.

# Download data and running

```
git clone https://github.com/fxia22/pointnet.pytorch
cd pointnet.pytorch
pip install -e .
```

Training 
```
cd utils
python train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
python train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```

Use `--feature_transform` to use feature transform.

# Performance

## Classification performance

On ModelNet40:



# Links

- [Project Page](http://stanford.edu/~rqi/pointnet/)
