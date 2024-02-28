from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
import torch

from typing import Union

class PTARLDataset(Dataset):

    def __init__(self, data, labels, continuous_cols, category_cols, label_class: Union[torch.FloatTensor, torch.LongTensor] = torch.LongTensor):
        super().__init__()
        self.labels = label_class(labels.values)
        self.columns = data.columns
        
        self.continuous_cols = continuous_cols
        self.category_cols = category_cols
        
        self.x_conts = torch.FloatTensor(data[continuous_cols].values)
        self.x_cats = torch.LongTensor(data[category_cols].values)
        
        if label_class == torch.LongTensor:
            class_counts = [sum((self.labels == i)) for i in set(self.labels.numpy())]
            num_samples = len(self.labels)

            class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
            self.weights = [class_weights[self.labels[i]] for i in range(int(num_samples))]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):

        x = self.x_conts[idx], self.x_cats[idx]
        y = self.labels[idx]

        return x, y
    

class PTARLDataModule(LightningDataModule):
    def __init__(self, train_ds, val_ds, test_ds, batch_size, n_jobs = 32):
        super().__init__()

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def setup(self, stage: str):
        sampler = WeightedRandomSampler(self.train_ds.weights, num_samples = len(self.train_ds))

        self.train_dl = DataLoader(self.train_ds, 
                                   batch_size = self.batch_size, 
                                   shuffle=False, 
                                   sampler = sampler,
                                   num_workers=self.n_jobs,
                                   drop_last=False)
        self.val_dl = DataLoader(self.val_ds, batch_size = self.batch_size, shuffle=False, sampler = SequentialSampler(self.val_ds), num_workers=self.n_jobs, drop_last=False)
        self.test_dl = DataLoader(self.test_ds, batch_size = self.batch_size, shuffle=False, sampler = SequentialSampler(self.test_ds), num_workers=self.n_jobs)
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl