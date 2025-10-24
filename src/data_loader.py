import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class Bangla_Handwriting_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])