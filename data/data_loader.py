import os
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import DataLoader as DL
import numpy as np
import ast
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as T
from .augmentations import VICRegAUgmentations


class nihdataset(torch.utils.data.Dataset):
    def __init__(self, df,class_names,transform):
        
        self.image_filepaths = df["filename"].values 
        self.transform = transform
        self.pathologies = class_names
        self.pathologies = sorted(self.pathologies)
        self.csv = df
        
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                mask = self.csv[pathology]
            self.labels.append(mask.values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        

    def __getitem__(self, idx):
        img = self.image_filepaths[idx]
        image = Image.open(img).convert('RGB')
        
        if self.transform:
            timage = self.transform(image)
            rbbox = lbbox = None
                
        label = self.labels[idx]
        return timage,label
    

    def __len__(self):
        return len(self.image_filepaths) 
    
           
class DataLoader():
    def __init__(self, config=None, model_name=None):
        self.config = config
        self.model_name      = model_name.lower() 
        self.batch_size      = config['data']['batch_size']
        self.data_workers    = config['data']['data_workers']
        self.tmode           = config['tmode']
        
        self.nih_train_df    = pd.read_csv(config['data']['nih']['train_df']) 
        self.nih_valid_df    = pd.read_csv(config['data']['nih']['valid_df']) 
        
        
        augmentations = {
                    'mlvicx':VICRegAUgmentations,
                    }
        
        if self.tmode == 'pre':
            transform_class      = augmentations[self.model_name]
            self.train_transform = transform_class(self.config)
            self.collate_fn = default_collate

            
        elif self.tmode == 'down':
            self.nih_box_df  = None
            self.collate_fn = default_collate
            self.train_transform = T.Compose([T.Resize((224, 224)), 
                                              T.RandomHorizontalFlip(p=0.5),
                                              T.RandomRotation(degrees=15), 
                                              T.ToTensor(),           
                                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                            ])
        else:
            print('select transformation mode out of down or pre')
            
        self.valid_augmentations = T.Compose([T.Resize((224, 224)), 
                                              T.ToTensor(),           
                                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                                            ]
    
    
    def GetNihDataset(self):
        train_df = self.nih_train_df.fillna(-1)
        valid_df = self.nih_valid_df.fillna(-1)
        class_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
           'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
           'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
                
        train_transform = self.train_transform
        valid_transform = self.valid_augmentations
        
        
        
        train_set = nihdataset(train_df,
                               class_names,
                               train_transform,
                               )
        valid_set = nihdataset(valid_df,
                               class_names,
                               valid_transform,
                               )
        train_loader = DL(dataset=train_set,
                         batch_size=self.batch_size,
                         collate_fn=self.collate_fn,
                         shuffle= True,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        
        valid_loader = DL(dataset=valid_set,
                         batch_size=self.batch_size,
                         collate_fn=self.collate_fn,
                         shuffle= False,
                         num_workers=self.data_workers,
                         pin_memory=True,
                         drop_last=True)
        print(f'{len(train_set)} images have loaded for training')
        print(f'{len(valid_set)} images have loaded for validation')
        
        return train_loader, valid_loader, class_names

            
