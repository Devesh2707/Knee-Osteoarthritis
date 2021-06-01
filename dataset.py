import os
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np

import torch
from torch import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import pytorch_lightning as pl

from PIL import Image

from utils import preprocess_image

from config import CONFIG

cfg = CONFIG()

dfx = pd.read_csv(cfg.trainset)

class Data(Dataset):
    def __init__(self, dataframe, transform, one_hot = False, use_preprocess = True):
        super().__init__()
        self.dataframe = dataframe
        self.transform = transform
        self.one_hot = one_hot
        self.use_preprocess = use_preprocess
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, item):
        img_path = self.dataframe.iloc[item]['image_path']
        target = self.dataframe.iloc[item]['label']
        image = Image.open(img_path).convert('RGB')
        image = np.asarray(image)
        if self.use_preprocess :
          image = preprocess_image(image)
        if self.transform is not None:
            image = self.transform(image = image)['image']
        if self.one_hot:
            return image, F.one_hot(torch.tensor(target), num_classes=cfg.num_classes)
        else:
            return image, torch.tensor(target)

class TestData(Dataset):
    def __init__(self, dataframe, transform, use_preprocess = True):
        super().__init__()
        self.dataframe = dataframe
        self.transform = transform
        self.use_preprocess = use_preprocess
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, item):
      img_id = self.dataframe.iloc[item]['filename']
      img_path = os.path.join(cfg.test_images,img_id)
      image = Image.open(img_path).convert('RGB')
      image = np.asarray(image)
      if self.use_preprocess:
        image = preprocess_image(image)
      if self.transform is not None:
        image = self.transform(image = image)['image']

      return image

class DataModule(pl.LightningDataModule):

    def __init__(self, fold: int, train_batch_size: int, valid_batch_size: int, one_hot: bool = False):
        super().__init__()
        self.fold = fold
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.one_hot = one_hot
        self.train_transform = cfg.train_aug
        self.valid_transform = cfg.val_aug

    def setup(self, stage: str = None):
        df_train = dfx[dfx.kfold != self.fold].reset_index(drop=True)
        df_valid = dfx[dfx.kfold == self.fold].reset_index(drop=True)

        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        self.train_dataset = Data(
            dataframe = df_train,
            transform = self.train_transform,
            one_hot = self.one_hot,
            use_preprocess = cfg.use_preprocess
        )

        self.valid_dataset = Data(
            dataframe = df_valid,
            transform = self.valid_transform,
            one_hot = self.one_hot,
            use_preprocess = cfg.use_preprocess
        )     

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch_size, shuffle = True, pin_memory = True, num_workers = 4, drop_last=True)
      
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size = self.valid_batch_size, shuffle = False, pin_memory = True, num_workers = 4, drop_last=True)