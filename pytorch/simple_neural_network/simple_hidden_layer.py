#! /Users/mahe/anaconda3/bin/python

import numpy as np
import torch
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt


class Net(nn.Module):
    def __init__(self, n_x, n_h, n_y):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_x, n_h)
        self.linear2 = nn.Linear(n_h, n_y)

    def forward(self, x):
        x = sigmoid(self.linear1(x))
        x = sigmoid(self.linear2(x))
        return x


def train(X, Y, model, optimizer, criterion, epochs=10000):
    cross_cost = []
    for epoch in range(epochs):
        total_cost = 0
        summary = 0
        for x, y in zip(X, Y):
            y_ = model(x)
            loss = criterion(y, y_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_cost += loss.item()
            summary += ((y_ > 0.5) == y).sum().item()
        cross_cost.append(total_cost)
        if epoch % 100 == 0:
            pass
            print('epoch {} is finished, cost is {}, predcit is {}'.
                  format(epoch, total_cost, summary/Y.shape[0]))
    return cross_cost


X = torch.arange(-2.0, 2.0, 0.1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0]).view(-1, 1)
Y[(X.data[:, 0] > -1.0) * (X.data[:, 0] < 1.)] = 1

criterion = nn.MSELoss()
model = Net(1, 3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
cross_cost = train(X, Y, model, optimizer, criterion, epochs=500)
plt.plot(cross_cost)
plt.xlabel('epochs')
plt.title('cross entropy loss')
plt.show()
