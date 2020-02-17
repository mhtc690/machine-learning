#! /Users/mahe/anaconda3/bin/python

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pylab as plt


class CNN(nn.Module):
    def __init__(self, out_1=16, out_2=32, num_classes=10):
        super(CNN, self).__init__()
        # first CNN layer
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1,
                              kernel_size=5, stride=1, padding=2)
        # Batch norm of first CNN layer
        self.cnn1_bn = nn.BatchNorm2d(out_1)
        # first Max Pooling layer
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # second CNN layer
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2,
                              kernel_size=5, stride=1, padding=2)
        # Batch norm of second CNN layer
        self.cnn2_bn = nn.BatchNorm2d(out_2)
        # Second max pooling layer
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer, 4*4 is item size
        self.fc = nn.Linear(out_2*4*4, num_classes)
        self.fc_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.max_pool1(torch.relu(self.cnn1_bn(self.cnn1(x))))
        x = self.max_pool2(torch.real(self.cnn2_bn(self.cnn2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc_bn(self.fc(x))
        return x


def train(model, optimizer, criterion, train_loader, test_loader, epochs):
    cost = []
    accuracies = []
    ret_result = {}
    for epoch in range(epochs):
        print('Epoch {} starts.'.format(epoch))
        for i, (x, y) in enumerate(train_loader):
            model.train()
            y_ = model(x)
            loss = criterion(y_, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost.append(loss.item())
            # print('Iter {} finished, loss is {}'.format(i, loss.item()))

        correct = 0
        for x_test, y_test in test_loader:
            model.eval()
            y_test_ = model(x_test)
            _, yhat = torch.max(y_test_.data, 1)
            correct += (y_test == yhat).sum().item()
        accuracy = correct / len(test_loader.dataset)
        accuracies.append(accuracy)
        print('Epoch {} finished, test accuracy {}%'.format(epoch, accuracy))

    ret_result['cost'] = cost
    ret_result['accuracy'] = accuracies
    return ret_result


# prepare train/test data set
IMAGE_SIZE = 16
composed = transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
        )
train_dset = dsets.MNIST('./', train=True, download=True, transform=composed)
test_dset = dsets.MNIST('./', train=False, download=True, transform=composed)
train_loader = torch.utils.data.DataLoader(dataset=train_dset, batch_size=256)
test_loader = torch.utils.data.DataLoader(dataset=test_dset, batch_size=4096)

# prepare for training
criterion = nn.CrossEntropyLoss()
model = CNN(out_1=8, out_2=16, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
result = train(model, optimizer, criterion, train_loader,
               test_loader, epochs=10)
