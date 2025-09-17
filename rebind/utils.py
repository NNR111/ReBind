import os
import h5py
import numpy as np
from torch.utils.data import Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class EditDistanceDataset(Dataset):
    def __init__(self, file_path, padded_length):
        self.data = []
        self.padded_length = padded_length
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                entry = f[key]
                original = np.array(entry['original'])
                edited = np.array(entry['edited'])
                distance = np.array(entry['distance'])
                self.data.append({
                    'original': np.pad(original, (0, padded_length - len(original)), 'constant'),
                    'edited': np.pad(edited, (0, padded_length - len(edited)), 'constant'),
                    'distance': distance
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return (torch.tensor(sample['original'], dtype=torch.float32),
                torch.tensor(sample['edited'], dtype=torch.float32),
                torch.tensor(sample['distance'], dtype=torch.float32))

class TripletDistanceDataset(Dataset):
    def __init__(self, file_path, padded_length):
        self.data = []
        self.padded_length = padded_length
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                group = f[key]
                # Read sequences
                anchor = np.array(group['anchor'])
                positive = np.array(group['positive'])
                negative = np.array(group['negative'])
                # Read distances
                d_anchor_positive = group.attrs['d_anchor_positive']
                d_anchor_negative = group.attrs['d_anchor_negative']
                d_positive_negative = group.attrs['d_positive_negative']

                # Pad sequences to a fixed length
                anchor = np.pad(anchor, (0, self.padded_length - len(anchor)), 'constant')
                positive = np.pad(positive, (0, self.padded_length - len(positive)), 'constant')
                negative = np.pad(negative, (0, self.padded_length - len(negative)), 'constant')

                self.data.append({
                    'anchor': anchor,
                    'positive': positive,
                    'negative': negative,
                    'd_anchor_positive': d_anchor_positive,
                    'd_anchor_negative': d_anchor_negative,
                    'd_positive_negative': d_positive_negative
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        anchor = torch.tensor(sample['anchor'], dtype=torch.float32)
        positive = torch.tensor(sample['positive'], dtype=torch.float32)
        negative = torch.tensor(sample['negative'], dtype=torch.float32)
        d_anchor_positive = torch.tensor(sample['d_anchor_positive'], dtype=torch.float32)
        d_anchor_negative = torch.tensor(sample['d_anchor_negative'], dtype=torch.float32)
        d_positive_negative = torch.tensor(sample['d_positive_negative'], dtype=torch.float32)

        return anchor, positive, negative, d_anchor_positive, d_anchor_negative, d_positive_negative

def load_triplet_datasets(dataset_folder, K_values, num_sequences, padding_length):
    datasets = []
    for K in K_values:
        file_path = os.path.join(dataset_folder, f"triplet_datasets_K_{K}_num_seq_{num_sequences}.h5")
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}")
            datasets.append(TripletDistanceDataset(file_path, padding_length))
        else:
            print(f"Dataset for K={K} and num. seq={num_sequences} not found.")        
    return datasets

def load_small_datasets(dataset_folder, K_values, padding_length):
    datasets = []
    for K in K_values:
        file_path = os.path.join(dataset_folder, f"edit_distance_dataset_K_{K}.h5")        
        if os.path.exists(file_path):
            datasets.append(EditDistanceDataset(file_path, padding_length))
            print("loaded dataset called edit_distance_dataset_K_{K}.h5")
        else:
            print(f"Dataset for K={K} not found.")        
    return datasets


def load_datasets(dataset_folder, K_values, padding_length):
    datasets = []
    for K in K_values:
        file_path = os.path.join(dataset_folder, f"edit_distance_dataset_K_{K}_10E6.h5")
        if os.path.exists(file_path):
            datasets.append(EditDistanceDataset(file_path, padding_length))
        else:
            print(f"Dataset for K={K} not found.")        
    return datasets

def train_test_split(datasets, test_ratio=0.2):
    train_datasets = []
    test_datasets = []
    for dataset in datasets:
        train_size = int((1 - test_ratio) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    return train_datasets, test_datasets


class H5TripletDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5 = h5py.File(h5_path, "r")
        # Flexible: support bytes keys or str keys
        self.anchor = self.h5["anchor"]
        self.positive = self.h5["positive"]
        self.negative = self.h5["negative"]
        self.d_ap = self.h5["d_ap"]
        self.d_an = self.h5["d_an"]
        self.d_pn = self.h5["d_pn"]
        self.length = self.anchor.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        a = torch.from_numpy(self.anchor[idx])
        p = torch.from_numpy(self.positive[idx])
        n = torch.from_numpy(self.negative[idx])
        d_ap = torch.tensor(self.d_ap[idx])
        d_an = torch.tensor(self.d_an[idx])
        d_pn = torch.tensor(self.d_pn[idx])
        return a, p, n, d_ap, d_an, d_pn


def load_data_triplets_from_h5(h5_path: str, batch_size: int):
    ds = H5TripletDataset(h5_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader
