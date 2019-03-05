import torch
from utils import load_data
from layer import GraphConvolution
import torch.optim as optim

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
opt = optim.Adam(model.parameters(), lr=0.01)

result = torch.from_numpy(y_train[train_mask]).float()
result_val = torch.from_numpy(y_val[val_mask]).float()
result_test = torch.from_numpy(y_test[test_mask]).float()
train_mask = torch.from_numpy(train_mask.astype(int)).nonzero().view(-1)
val_mask = torch.from_numpy(val_mask.astype(int)).nonzero().view(-1)
test_mask = torch.from_numpy(test_mask.astype(int)).nonzero().view(-1)

loss_window = [0, 0]
loss_inc = 0

for epoch in range(10):
    torch.set_grad_enabled(True)
    output = model(Features)
    loss = torch.nn.MSELoss(output[train_mask], result)
    print(loss)
    # loss = - (result * torch.log(output[train_mask]+1e-10)).sum()

    opt.zero_grad()
    loss.backward()
    opt.step()

    train_acc = (output[train_mask].argmax(dim=1) ==
            result.argmax(dim=1)).sum() / result.shape[0]
    print('epochs: {}, train loss: {}, train accuracy: {}'.format(
        (epoch+1), loss, train_acc))

    torch.set_grad_enabled(False)
    loss_val = - (result_val * torch.log(output[val_mask]+1e-10)).sum()
    val_acc = (output[val_mask].argmax(dim=1) ==
            result_val.argmax(dim=1)).sum() / result_val.shape[0]
    print(output[train_mask].argmax(dim=1))
    print('val loss: {}, val accuracy: {}'.format(loss_val, val_acc))
    if loss_val > loss_window[1]:
        loss_inc += 1
    else:
        loss_inc = 0
    if loss_inc >= 10:
        break
    loss_window.append(loss_val)
    loss_window.pop(0)

test_acc = (output[test_mask].argmax(dim=1) ==
        result_test.argmax(dim=1)).sum() / result_test.shape[0]
print('test accuracy: {}'.format(test_acc))
