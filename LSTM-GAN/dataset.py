import torch
import numpy as np

class BondsDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        signal = self.data[index]
     
        return torch.tensor(self.normalize(np.float32(signal)))
    
    def normalize(self,x):
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min())/(x.max() - x.min()) - 1)
    
    def denormalize(self, x):
        return 0.5 * (x*self.max - x*self.min + self.max + self.min)