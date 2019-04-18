import torch
import numpy as np
import pickle
# data loader
class TrainDatasetFromPKL:
    """ Load dataset from .npy file """

    def __init__(self, path,  transform=None):
        """
        Args:
            root_dir (string): the .npy file .
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(path,"rb") as file:
            self.x,self.y=pickle.load(file)
        self.x=self.x.transpose(0,3,1,2)
        self.x=self.x.astype(np.float32)
        print(self.x.shape)
        # self.HRData = np.load(hr_dir)[:train_num, :, :, :]
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]