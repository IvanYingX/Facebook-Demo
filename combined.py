import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from image_text_loader import ImageTextDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import pickle

# Image model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Text model
class TextClassifier(nn.Module):
    def __init__(self,
                 ngpu,
                 input_size: int = 768):
        super(TextClassifier, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(nn.Conv1d(input_size, 256, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2, stride=2),
                                  nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(192 , 128))

    def forward(self, inp):
        x = self.main(inp)
        return x

# Combine models

class CombinedModel(nn.Module):
    def __init__(self,
                 ngpu,
                 input_size: int = 768,
                 num_classes: int = 2):
        super(CombinedModel, self).__init__()
        self.ngpu = ngpu
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        out_features = resnet50.fc.out_features
        self.image_classifier = nn.Sequential(resnet50, nn.Linear(out_features, 128)).to(device)
        self.text_classifier = TextClassifier(ngpu=ngpu, input_size=input_size)
        self.main = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, image_features, text_features):
        image_features = self.image_classifier(image_features)
        text_features = self.text_classifier(text_features)
        combined_features = torch.cat((image_features, text_features), 1)
        combined_features = self.main(combined_features)
        return combined_features


dataset = ImageTextDataset()
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
combined = CombinedModel(ngpu=1, input_size=768, num_classes=dataset.num_classes)
combined.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined.parameters(), lr=0.001)
epochs = 5
for epoch in range(epochs):
    hist_acc = []
    hist_loss = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (image_features, text_features, labels) in pbar:
        image_features = image_features.to(device)
        text_features = text_features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = combined(image_features, text_features)
        loss = criterion(outputs, labels)
        loss.backward()
        hist_acc.append(torch.mean((torch.argmax(outputs, dim=1) == labels).float()).item())
        hist_loss.append(loss.item())
        optimizer.step()
        pbar.set_description(f'Epoch {epoch + 1}/{epochs} Loss: {loss.item():.4f} Acc = {round(torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels), 2)} Total_acc = {round(np.mean(hist_acc), 2)}')

torch.save(combined.state_dict(), 'combined_model.pt')

with open('combined_decoder.pkl', 'wb') as f:
    pickle.dump(dataset.decoder, f)
# print(image_classifier(next(iter(train_loader)).to(device)))
