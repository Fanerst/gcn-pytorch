import torch
from torch import nn
import numpy as np

default_dtype_torch = torch.float32

class ConvLayer(nn.Linear):
    """ graph convolution layer """
    def __init__(self, D, A, bias):
        super(ConvLayer, self).__init__(D, D, bias)
        self.A = A

    def forward(self, input):
        temp = torch.sparse.mm(self.A.to_sparse(), input)
        output = nn.functional.linear(temp, self.weight, self.bias)
        return output

class GraphConvolution(nn.Module):
    """ graph convolution network """
    def __init__(self, N, D, net_depth, A):
        super().__init__()
        """
        N is the number of nodes in the graph
        D is the number of features
        net_depth is the depth of the network
        A is the normalized laplacian matrix
        """

        self.net = []
        for l in range(net_depth-1):
            self.net.extend([ConvLayer(D, A, 1), nn.Sigmoid()])
        self.net.pop()
        self.net.extend([nn.Softmax()])
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        output = self.net(input)
        return output
