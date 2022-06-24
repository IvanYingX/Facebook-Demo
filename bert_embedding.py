from text_loader_bert import TextDatasetBert
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import pickle

class Classifier(nn.Module):
    def __init__(self,
                 ngpu,
                 input_size: int = 768,
                 num_classes: int = 2):
        super(Classifier, self).__init__()
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
                                  nn.Linear(384 , 128),
                                  nn.ReLU(),
                                  nn.Linear(128, num_classes))
    def forward(self, inp):
        x = self.main(inp)
        return x


dataset = TextDatasetBert(labels_level=0, max_length=100)
ngpu = 2
n_epochs = 5
batch_size = 32
num_classes = dataset.num_classes
dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=1)                                
lr = 0.001
criterion = nn.CrossEntropyLoss()

clf = Classifier(num_classes=num_classes, ngpu=ngpu)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
clf.to(device)
optimizer = optim.Adam(clf.parameters(), lr=lr)

losses = []

for epoch in range(n_epochs):
    hist_accuracy = []
    accuracy = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (data, labels) in pbar:
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = clf(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accuracy = torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels)
        hist_accuracy.append(accuracy)
        losses.append(loss.item())
        pbar.set_description(f"Epoch = {epoch+1}/{n_epochs}. Acc = {round(torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels), 2)}, Total_acc = {round(np.mean(hist_accuracy), 2)}, Losses = {round(loss.item(), 2)}" )

torch.save(clf.state_dict(), 'model_bert.pt')

with open('text_decoder.pkl', 'wb') as f:
    pickle.dump(dataset.decoder, f)
