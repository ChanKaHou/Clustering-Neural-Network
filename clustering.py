'''
Clustering program source code
for research article "Probability k-means Clustering for Neural Network Architecture"

Version 1.0
(c) Copyright 2019 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The Clustering program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The Clustering program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
import random

clustering = 15

class Net(torch.nn.Module):
    def __init__(self, data):
        super(Net, self).__init__()
        self.pred = torch.nn.Parameter(torch.normal(torch.zeros(clustering, 2) + data.mean(dim=0), 0.01))

    def forward(self, data):
        distance = torch.sum((self.pred - data)**2, dim=-1).sqrt() #torch.Size([batch, clustering])
        #prob = torch.nn.functional.softmax(distance.reciprocal())
        prob = torch.nn.functional.normalize(distance.reciprocal(), dim=-1)**2
        return prob

    def get(self):
        return self.pred

def clusterLoss(prob, data):
    prob = prob.unsqueeze(2) #torch.Size([batch, clustering, 1])
    sumProb = prob.sum(0) #torch.Size([clustering, 1])

    probData = prob*data #torch.Size([batch, clustering, 2])
    sumProbData = probData.sum(0) #torch.Size([clustering, 2])

    probMean = sumProbData/sumProb #torch.Size([clustering, 2])

    probLoss = ((data - probMean)**2)*prob
    sumProbLoss = probLoss.sum(0)

    loss = sumProbLoss/sumProb
    return loss.sum()

data = torch.tensor(numpy.loadtxt(fname = "s2.txt")).type(torch.FloatTensor)/1000000
net = Net(data)
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

while True:
    prob = net(data.unsqueeze(1))

    loss = clusterLoss(prob, data.unsqueeze(1))
    #print(loss)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    plt.cla()
    label = torch.max(prob, -1)[1]
    plt.scatter(data[:,0], data[:,1], s=3, c=label, cmap=plt.cm.get_cmap('tab20'))
    pred = net.get().data.numpy()
    plt.scatter(pred[:,0], pred[:,1], c="black")
    plt.pause(0.001)
