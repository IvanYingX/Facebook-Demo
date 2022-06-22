import torch
import os
import pandas as pd

class TextDataset(torch.utils.data.Dataset):
    '''
    The TextDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all descriptions from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transform the images so they can be used in a model
    Parameters
    ----------
    root_dir: str
        The directory of the CSV with the products

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
                 root_dir: str = 'cleaned_products.csv'):
        
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The file {self.root_dir} does not exist")
        products = pd.read_csv(self.root_dir, lineterminator='\n')
        products['category'] = products['category'].apply(lambda x: self.get_category(x, labels_level))
        self.labels = products['category'].to_list()
        self.descriptions = products['product_description'].to_list()

        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}

    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        # label = torch.as_tensor(label)
        description = self.descriptions[index]
        return description, label

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def get_category(x, level: int = 0):
        return x.split('/')[level].strip()

if __name__ == '__main__':
    dataset = TextDataset(labels_level=0)
    print(dataset.num_classes)
    print(dataset[0], dataset.decoder[dataset[0][1]])


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=12,
                                             shuffle=True, num_workers=1)
    for i, (data, labels) in enumerate(dataloader):
        print(data)
        print(labels)
        if i == 0:
            break
