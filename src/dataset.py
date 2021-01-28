import os
from torch import tensor
import torchvision
from PIL import Image 
from torch.utils.data import Dataset

class LeafData(Dataset):
    def __init__(self, df, dir):
        self.df = df
        self.dir = dir
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        img_id, label, _ = self.df.loc[item]
        img_path = os.path.join(self.dir, img_id)
        image = Image.open(img_path).resize((256,256))
        return {'image': torchvision.transforms.ToTensor()(image), 'label':tensor(label)}