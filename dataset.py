from torch.utils.data import Dataset
import numpy as np

class PointCloudData(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.points = np.load(dataset_dir+'coors.npy')
        self.labels= np.load(dataset_dir+'labels.npy')
        self.transforms = transform 
    
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.points[idx])
        return {'pointcloud': pointcloud, 
                'category': self.labels[idx]}

