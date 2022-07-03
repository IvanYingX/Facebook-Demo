
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import requests
import random
import torch
import os
import random
import json
import pandas as pd


ImageFile.LOAD_TRUNCATED_IMAGES = True


def repeat_channel(x):
            return x.repeat(3, 1, 1)

class ImageDataset(torch.utils.data.Dataset):
    '''
    The ImageDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transform the images so they can be used in a model
    Parameters
    ----------
    root_dir: str
        The directory with subfolders containing the images
    transform: torchvision.transforms 
        The transformation or list of transformations to be 
        done to the image. If no transform is passed,
        the class will do a generic transformation to
        resize, convert it to a tensor, and normalize the numbers
    
    Attributes
    ----------
    files: list
        List with the directory of all images
    labels: set
        Contains the label of each sample
    encoder: dict
        Dictionary to translate the label to a 
        numeric value
    decoder: dict
        Dictionary to translate the numeric value
        to a label
    '''

    def __init__(self,
                 labels_level: int = 0,
                 transform: transforms = None,
                 merge: bool = False,
                 download: bool = False):
        
        if download:
            self.download()
        else:
            if not os.path.exists('cleaned_images'):
                raise RuntimeError('Image Dataset not found, use download=True to download it')

        if merge:
            self.products = self.merge()
        else:
            self.products = pd.read_csv('product_images.csv', lineterminator='\n')

        self.products['category'] = self.products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = self.products['category'].to_list()

        # Get the images        
        self.files = self.products['image_id']
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open('cleaned_images/' + self.files[index] + '.jpg')
        if image.mode != 'RGB':
          image = self.transform_Gray(image)
        else:
          image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.files)

    @staticmethod
    def merge():
        products = pd.read_csv('cleaned_products.csv', lineterminator='\n')
        images = pd.read_csv('Images.csv')
        products_images = products.merge(images, left_on='id', right_on='product_id').rename(columns={'id_y': 'image_id'}).drop('id_x', axis=1)
        products_images.to_csv('product_images.csv')
        return products_images

    def download(self):

        pass
    
    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()


def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    return train_dataset, validation_dataset

if __name__ == '__main__':
    dataset = ImageDataset(merge=True)
    print(dataset[0][0])
    print(dataset.decoder[int(dataset[0][1])])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
                                             shuffle=True, num_workers=1)
    for i, (data, labels) in enumerate(dataloader):
        print(data)
        print(labels)
        print(data.size())
        if i == 0:
            break