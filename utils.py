import torch
import torchvision
from dataset import PointCloudData
from torch.utils.data import DataLoader

# Normalize the pointcloud to a unit sphere
"""
    Pointcloud data will now have zero mean and will be centered around the origin, normalized to a unit sphere.
    pointcloud is of shape (n,3) where n is the number of points
"""
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
#Rotation
"""
    Function to randomly rotate a pointcloud (object) by an angle theta along the z axis, this is useful to make a deep learning model view invariant.
    Explanation can be found at https://en.wikipedia.org/wiki/Rotation_matrix #Basic_rotations.
    pointcloud is of shape (n,3) where n is the number of points
"""
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
#Adding noise
"""
    To make a deep learning model robust to variances in distributions of input data, a slight noise is added to the original pointcloud.
    Jitter the position of each points by a Gaussian noise with zero mean and 0.02 standard deviation.
"""
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class shuffle(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        # shuffle points
        #This function only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.
        np.random.shuffle(pointcloud)
        shuffle_pointcloud = pointcloud
        return shuffle_pointcloud
    
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)
    
def get_loaders(dataset_dir,batch_size,train_transforms,test_transforms):
    train_ds = PointCloudData(
        dataset_dir=dataset_dir,
        transform=train_transforms,
    )

    train_loader = DataLoader(
        train_ds+'train/',
        batch_size=batch_size,
        shuffle=True,
    )

    test_ds = PointCloudData(
        dataset_dir=dataset_dir+'test/',
        transform=test_transforms,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader

#The loss function is implemented based on the PointNet paper.
#As we used LogSoftmax for stability, we should apply NLLLoss instead of CrossEntropyLoss.
#Also, we will add two regularization terms in order transformations matrices to be close to orthogonal ( AAᵀ = I )
def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)