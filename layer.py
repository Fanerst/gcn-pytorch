import torch
from torch import nn
import numpy as np

default_dtype_torch = torch.float32

class ConvLayer(nn.Linear):
    """ graph convolution layer """
    def __init__(self, in_features, out_features,A, bias):
        super(ConvLayer, self).__init__(in_features, out_features, bias)
        self.A = A

    def forward(self, input):
        temp = torch.sparse.mm(self.A.to_sparse(), input)
        output = nn.functional.linear(temp, self.weight, self.bias)
        return output

class GraphConvolution(nn.Module):
    """ graph convolution network """
    def __init__(self, N, C, H, F, net_depth, A):
        super().__init__()
        """
        N is the number of nodes in the graph
        C is the number of classes
        H is the hidden dimension
        F is the number of features
        net_depth is the depth of the network
        A is the normalized laplacian matrix
        """

        self.net = []
        hd = [F] + [H]*(net_depth-1) + [C]
        for l in range(net_depth):
            self.net.extend([ConvLayer(hd[l], hd[l+1], A, 1), nn.Sigmoid()])
        self.net.pop()
        self.net.extend([nn.Softmax(dim=1)])
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        output = self.net(input)
        return output
