from dataclasses import dataclass
from functools import partial
import random
from typing import Tuple, Union

from PIL import ImageFilter, Image
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import ImageFilter
import copy
import numpy as np
from torchvision.transforms import InterpolationMode

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

       
class VICRegAUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=1.0),
                T.RandomSolarize(threshold=128, p=0.0),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
        self.augment_prime = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.1),
                T.RandomSolarize(threshold=128, p=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
        
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment_prime(x)
        return [x1,x2]


