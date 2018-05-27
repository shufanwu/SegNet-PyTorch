import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

from lcn import LocalContrastNormalization

raw_images_path = './CamVid600/raw_images/'
mask_path = './CamVid600/mask/'

transform = transforms.Compose([
    #transforms.Resize((360, 480)),
    transforms.ToTensor()
])
class CamVidDataset(Dataset):
    def __init__(self, phase='train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.raw_images = os.listdir(os.path.join(raw_images_path,self.phase))

    def __getitem__(self, idx):
        image = Image.open(os.path.join(raw_images_path,self.phase,self.raw_images[idx])).convert('RGB')
        image_name = self.raw_images[idx].split('.')[0]
        #label_name = f'{image_name}_L.png'
        label_name = f'{image_name}.png'
        label = Image.open(os.path.join(mask_path,self.phase,label_name)).convert('L')
        #label = label.resize((480,360))
        image = transform(image)
        LCN = LocalContrastNormalization()
        image = LCN(image)
        label = np.array(label, dtype=int)
        return image, label

    def __len__(self):
        return len(self.raw_images)