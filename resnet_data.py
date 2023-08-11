import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SimulatorDataset(Dataset):
    def __init__(self, data_path, phase='train'):
        self.folder_dir = os.path.dirname(data_path) # get the root of image folder
        # extract the image name and labels into a list
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(list(line.strip('\n').split(' ')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_p, a, b, c, d = self.data[idx]  # image path, throttle, brake, steering angle, velocity
        img_path = os.path.join(self.folder_dir, img_p)
        a = int(a)
        b = int(b)
        c = int(c)
        label = [a, b, c]

        return img_path, label

