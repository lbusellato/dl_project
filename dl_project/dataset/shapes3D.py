import h5py
import numpy as np
import random

from os import mkdir
from os.path import exists, dirname, abspath, join
from torch.utils.data import Dataset

ROOT = dirname(dirname(dirname(abspath(__file__))))

class Shapes3D(Dataset):
    
    def __init__(self,
                 subsample = 0.2,
                 max_pairs = 10) -> None:
        super().__init__()
        
        self.subsample = subsample
        self.max_pairs = max_pairs

        # Create the folder for cache storage
        self.cache_folder = join(ROOT, 'cache')
        if not exists(self.cache_folder):
            mkdir(self.cache_folder)

        # Download the dataset
        self.dataset_path = join(self.cache_folder, '3dshapes.h5')
        if not exists(self.dataset_path):
            raise FileNotFoundError('Download the dataset from https://storage.cloud.google.com/3d-shapes/3dshapes.h5 and place it in the cache folder.')
                
        # Generate the new dataset as described in the paper, if not done already
        self.images, self.labels, self.pairs = self.load_data()

    def load_data(self):

        if not exists(join(self.cache_folder, 'pairs.npy')):      
            # Convert the data to numpy arrays if they are not stored in the cache
            if not exists(join(self.cache_folder, 'images.npy')):
                with h5py.File(self.dataset_path, 'r') as f:
                    images = np.array(f['images'])
                    labels = np.array(f['labels'])
                    # Subsample the dataset
                    if self.subsample < 1.0:
                        indices = np.random.choice(len(images), int(len(images) * self.subsample), replace=False)
                        images = images[indices]
                        labels = labels[indices]
                    np.save(join(self.cache_folder, 'images.npy'), images)
                    np.save(join(self.cache_folder, 'labels.npy'), labels)
            else:
                images = np.load(join(self.cache_folder, 'images.npy'))
                labels = np.load(join(self.cache_folder, 'labels.npy'))
            
            # Create a list of indexes, by putting them together by scale, shape and orientation (columns 3,4,5)
            indices_by_attributes = {}
            for idx, (scale, shape, orientation) in enumerate(zip(labels[:, 3], labels[:, 4], labels[:, 5])):
                key = (scale, shape, orientation)
                if key not in indices_by_attributes:
                    indices_by_attributes[key] = []
                indices_by_attributes[key].append(idx)
            
            pairs = []
            for key, indices in indices_by_attributes.items():
                if len(indices) < 2:
                    continue
                # Create pairs within each group, limit the number of pairs to max_pairs_per_group
                for _ in range(self.max_pairs):
                    idx1, idx2 = random.sample(indices, 2)
                    if (idx1, idx2) not in pairs: pairs.append((idx1, idx2))
            
            np.save(join(self.cache_folder, 'pairs.npy'), pairs)
        else:
            images = np.load(join(self.cache_folder, 'images.npy'))
            labels = np.load(join(self.cache_folder, 'labels.npy'))
            pairs = np.load(join(self.cache_folder, 'pairs.npy'))
        return images, labels, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        image1, image2 = self.images[idx1], self.images[idx2]
        label1 = self.labels[idx1]
        label2 = self.labels[idx2]
        return (image1, label1), (image2, label2)
    