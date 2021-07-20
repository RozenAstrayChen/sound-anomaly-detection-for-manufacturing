import pandas as pd
import numpy as np
import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 

class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.target = label
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx], self.data[idx]
        