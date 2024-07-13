import h5py
import numpy as np
import random
import torch
import urllib

from dl_project.utils.custom_typing import Shapes3DData
from os import mkdir
from os.path import exists, dirname, abspath, join
from torch.utils.data import Dataset

ROOT = dirname(dirname(dirname(abspath(__file__))))

class Shapes3D(Dataset):
    
    def __init__(self,
                 max_pairs = 500,
                 batch_size = 64) -> None:
        super().__init__()
        
        self.max_pairs = max_pairs
        self.batch_size = batch_size

        # Create the folder for cache storage
        self.cache_folder = join(ROOT, 'cache')
        if not exists(self.cache_folder):
            mkdir(self.cache_folder)

        # Download the dataset
        self.dataset_path = join(self.cache_folder, '3dshapes.h5')
        if not exists(self.dataset_path):
            urllib.urlretrieve('https://storage.googleapis.com/3d-shapes/3dshapes.h5', './cache/3dshapes.h5')
                            
        # Lookup values for converting between features and labels        
        self.hues = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.scales = [0.75, 0.82142857, 0.89285714, 0.96428571, 1.03571429, 1.10714286, 1.17857143, 1.25]
        self.shapes = [0, 1, 2, 3]
        self.orientations = [-30, -25.71428571, -21.42857143, -17.14285714, -12.85714286,
                            -8.57142857,  -4.28571429,   0,           4.28571429,   8.57142857,
                            12.85714286,  17.14285714,  21.42857143,  25.71428571,  30]

        # Generate the new dataset as described in the paper, if not done already
        self.images, self.labels, self.pairs = self.load_data()

    def convert_features_to_labels(self, features):
        def find_index_of_closest(arr, val):
            return np.argmin(np.abs(np.array(arr)-val))
        return np.stack([[find_index_of_closest(self.hues, f[0]),
                          find_index_of_closest(self.hues, f[1]),
                          find_index_of_closest(self.hues, f[2]),
                          find_index_of_closest(self.scales, f[3]),
                          find_index_of_closest(self.shapes, f[4]),
                          find_index_of_closest(self.orientations, f[5])] for f in features])

    def load_data(self):

        if not exists(join(self.cache_folder, 'pairs.npy')):    
            # Convert the data to numpy arrays if they are not stored in the cache
            if not exists(join(self.cache_folder, 'images.npy')):
                with h5py.File(self.dataset_path, 'r') as f:
                    images = np.array(f['images'])
                    labels = np.array(f['labels'])
                    # Convert from raw values to labels
                    labels = self.convert_features_to_labels(labels)
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
                max_attempts = 10
                for _ in range(self.max_pairs):
                    for _ in range(max_attempts):
                        idx1, idx2 = random.sample(indices, 2)
                        if (idx1, idx2) not in pairs and idx1 != idx2: 
                            pairs.append((idx1, idx2))
                            break
        
            np.save(join(self.cache_folder, 'pairs.npy'), pairs)
        else:
            images = np.load(join(self.cache_folder, 'images.npy'))
            labels = np.load(join(self.cache_folder, 'labels.npy'))
            pairs = np.load(join(self.cache_folder, 'pairs.npy'))
        return images, labels, pairs

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(self.pairs))
        idx1, idx2 = self.pairs[idx]
        image1, image2 = self.images[idx1], self.images[idx2]
        label1 = self.labels[idx1]
        label2 = self.labels[idx2]
        return Shapes3DData(
            x=torch.as_tensor(image1,device='cuda',dtype=torch.float32).permute(2,0,1),
            y=torch.as_tensor(image2,device='cuda',dtype=torch.float32).permute(2,0,1),
            x_floor_hue_label=torch.as_tensor(label1[0],device='cuda'),
            x_wall_hue_label=torch.as_tensor(label1[1],device='cuda'),
            x_object_hue_label=torch.as_tensor(label1[2],device='cuda'),
            y_floor_hue_label=torch.as_tensor(label2[0],device='cuda'),
            y_wall_hue_label=torch.as_tensor(label2[1],device='cuda'),
            y_object_hue_label=torch.as_tensor(label2[2],device='cuda'),
            scale_label=torch.as_tensor(label1[3],device='cuda'),
            shape_label=torch.as_tensor(label1[4],device='cuda'),
            orientation_label=torch.as_tensor(label1[5],device='cuda'),
        )

    