#! /Users/mahe/anaconda3/bin/python

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

torch.manual_seed(0)


class Dataset():
    def __init__(self, transform=None, train=None):
        directory = "./resources/data"
        negative = "Negative"
        positive = "Positive"
        negative_file_path = os.path.join(directory, negative)
        negative_files = [os.path.join(negative_file_path, file_name)
                          for file_name in os.listdir(negative_file_path)
                          if file_name.endswith('.jpg')]
        negative_files.sort()
        positive_file_path = os.path.join(directory, positive)
        positive_files = [os.path.join(positive_file_path, file_name)
                          for file_name in os.listdir(positive_file_path)
                          if file_name.endswith('.jpg')]
        positive_files.sort()

        total_samples = len(negative_files) + len(positive_files)
        total_files = [None] * total_samples
        total_files[::2] = negative_files
        total_files[1::2] = positive_files

        Y = torch.ones([total_samples]).type(torch.LongTensor)
        Y[::2] = 0
        Y[1::2] = 1

        if train:
            self.images = total_files[0:30000]
            self.Y = Y[0:30000]
        else:
            self.images = total_files[30000:]
            self.Y = Y[30000:]
        self.len = len(self.images)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        y = self.Y[idx]
        return image, y


class Sigmoid(nn.Module):
    def __init__(self, n_input, n_output):
        super(Sigmoid, self).__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        y = torch.sigmoid(self.linear(x.view(x.size(0), -1)))
        return y


def train(model, optimizer, criterion, train_loader, test_loader, epochs):
    cost = []
    train_acc_list = []
    test_acc_list = []
    ret_val = {}
    for epoch in range(epochs):
        train_result = 0
        i = 0
        for x, y in train_loader:
            model.train()
            y_ = model(x)
            loss = criterion(y_, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost.append(loss.item())
            _, y_ = torch.max(y_, 1)
            iter_result = (y_ == y).sum().item()
            train_result += iter_result
            i += 1
            print('train iter {}, loss {}, train_acc {}'.
                  format(i, loss.item(), iter_result/x.size(0)))
        train_acc = train_result / len(train_loader.dataset)
        train_acc_list.append(train_acc)

        test_result = 0
        for x_test, y_test in test_loader:
            model.eval()
            y_test_ = model(x_test)
            _, y_test_ = torch.max(y_test_, 1)
            test_result += (y_test_ == y_test).sum().item()
        test_acc = test_result / len(test_loader.dataset)
        test_acc_list.append(test_acc)
        print('after epoch {}, train accuracy {}, test accuracy {}'.
              format(epoch, train_acc, test_acc))

    ret_val['cost'] = cost
    ret_val['train_acc'] = train_acc_list
    ret_val['test_acc'] = test_acc_list
    return ret_val


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
data_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])
train_data = Dataset(train=True, transform=data_trans)
test_data = Dataset(train=False, transform=data_trans)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=256)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4096)
model = models.resnet18(pretrained=True)
for parameter in model.parameters():
    parameter.requires_grad = False
n_fc = model.fc.in_features
model.fc = nn.Linear(n_fc, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([parameter for parameter in model.parameters()
                             if parameter.requires_grad], lr=0.003)
ret_value = train(model, optimizer, criterion, train_loader, test_loader, 1)
