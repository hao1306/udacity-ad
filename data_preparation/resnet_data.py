import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SimulatorDataset(Dataset):
    def __init__(self, data_path, phase='train'):
        # self.folder_dir = os.path.dirname(data_path) # get the root of image folder
        # extract the image name and labels into a list
        self.folder_dir = os.path.dirname(data_path)
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(list(line.strip('\n').split(' ')))
        self.data = self.data[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_p, a, b, c, d, e, f, g = self.data[idx]  # image path, throttle, brake, steering angle, velocity
        img_p = img_p[1:]
        imgpath = os.path.join(self.folder_dir, img_p)
        # a = int(a)
        # b = int(b)
        # c = int(c)
        # label = [a, b, c]
        img = cv2.imread(imgpath)
        img = img.astype(np.float32)/255

        img = torch.from_numpy(img).permute(2,0,1)

        return img, torch.from_numpy(np.array([a, b, c]).astype(np.float32))

# path = 'data/simulator/Dataset/VehicleData.txt'
# data1 = SimulatorDataset(path)
# path2 = data1[1]
