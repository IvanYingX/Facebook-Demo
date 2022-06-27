import torchvision
import torch.nn as nn
from image_loader import ImageDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pickle

dataset = ImageDataset()
ngpu = 2
n_epochs = 1
batch_size = 32
lr = 0.001
num_classes = dataset.num_classes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# Image model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ImageClassifier(nn.Module):
    def __init__(self,
                 ngpu,
                 num_classes):
        super(ImageClassifier, self).__init__()
        self.ngpu = ngpu
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = self.resnet50.fc.out_features
        self.linear = nn.Linear(out_features, num_classes).to(device)
        self.main = nn.Sequential(self.resnet50, self.linear).to(device)
    
    def forward(self, inp):
        x = self.main(inp)
        return x

model_conv = ImageClassifier(ngpu=ngpu, num_classes=num_classes)
# print(model_conv.resnet50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.parameters(), lr=lr)

losses = []

for epoch in range(n_epochs):
    hist_accuracy = []
    accuracy = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (data, labels) in pbar:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_conv(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accuracy = torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels)
        hist_accuracy.append(accuracy)
        losses.append(loss.item())
        pbar.set_description(f"Epoch = {epoch+1}/{n_epochs}. Acc = {round(torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels), 2)}, Total_acc = {round(np.mean(hist_accuracy), 2)}, Losses = {round(loss.item(), 2)}" )
        # print(f"Epoch: {epoch} | Batch: {i} | Loss: {loss.item()}")
        # print('-'*20)
        # print(f"Accuracy: {torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels)}")
        # print('-'*20)

torch.save(model_conv.state_dict(), 'resnet_50.pt')

with open('image_decoder.pkl', 'wb') as f:
    pickle.dump(dataset.decoder, f)