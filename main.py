import torch
from utils import load_data
from layer import GraphConvolution

datasets = 'cora'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(datasets)

ADJ = torch.from_numpy(adj.todense()).float()
ADJ += torch.eye(ADJ.shape[0], dtype=torch.float32)
degree_matrix = torch.diag(torch.sum(ADJ, dim=0))
D = degree_matrix.sqrt().inverse()
A = torch.mm(torch.mm(D,ADJ), D)

Features = torch.from_numpy(features.todense()).float()
N = ADJ.shape[0]
C = y_train.shape[1]
H = 16
D = Features.shape[1]

model = GraphConvolution(N, C, H, D, 3, A)
data = torch.from_numpy(y_train[train_mask]).float()
data_val = torch.from_numpy(y_val[val_mask]).float()
data_test = torch.from_numpy(y_test[test_mask]).float()

result = torch.from_numpy(features[train_mask]).float()
result_val = torch.from_numpy(features[val_mask]).float()
result_test = torch.from_numpy(features[test_mask]).float()

for epoch in range(num_epochs):
    output = model(data)
    loss = torch.cross_entropy(output, result)

