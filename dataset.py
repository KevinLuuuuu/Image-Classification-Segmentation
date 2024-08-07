from torch.utils.data import Dataset
import os
from PIL import Image
import imageio
import numpy as np
import torch

class ImageDataset(Dataset):

    def __init__(
        self,
        data_path,
        transform, 
        train_set = True
    ):
        super(ImageDataset).__init__()
        self.data_path = data_path
        self.images = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.endswith(".png")]
        self.transform = transform
        self.train_set = train_set
  
    def __len__(self):
        return len(self.images)
  
    def __getitem__(self,index):
        if self.train_set:
            image_name = self.images[index]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            try:
                label = int(image_name.split("/")[-1].split("_")[0])
            except:
                label = -1
            return image, label
        else:
            image_name = self.images[index]
            image = Image.open(image_name)
            if self.transform is not None:
                image = self.transform(image)
            image_name = image_name.split('/')[-1]

            return image, image_name


class SegmentationDataset(Dataset):

    def __init__(
        self,
        data_path,
        transform,
        train_set = True
    ):
        super(SegmentationDataset).__init__()
        self.data_path = data_path
        self.images = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.endswith(".jpg")]
        self.transform = transform
        self.train_set = train_set
  
    def __len__(self):
        return len(self.images)
  
    def __getitem__(self,index):
        sat_name = self.images[index]
        mask_name = sat_name.replace("sat.jpg", "mask.png")

        if self.train_set == True:
            sat_image = Image.open(sat_name)
            mask_image = np.zeros((7, 512, 512))
            mask = imageio.imread(mask_name)
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

            mask_image[0][mask == 3] = 1  # (Cyan: 011) Urban land 
            mask_image[1][mask == 6] = 1  # (Yellow: 110) Agriculture land 
            mask_image[2][mask == 5] = 1  # (Purple: 101) Rangeland 
            mask_image[3][mask == 2] = 1  # (Green: 010) Forest land 
            mask_image[4][mask == 1] = 1  # (Blue: 001) Water 
            mask_image[5][mask == 7] = 1  # (White: 111) Barren land 
            mask_image[6][mask == 0] = 1  # (Black: 000) Unknown 
            
            if self.transform is not None:
                sat_image = self.transform(sat_image)
            mask_image = torch.from_numpy(mask_image)

            return sat_image, mask_image
        
        else:
            sat_image = Image.open(sat_name)
            test_mask_name = "/" + sat_name.split('/')[-1]
            test_mask_name = test_mask_name.replace("sat.jpg", "mask.png")
            #print(test_mask_name)
            if self.transform is not None:
                sat_image = self.transform(sat_image)
            return sat_image, test_mask_name