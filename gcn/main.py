import torch
from utils import load_data
from layer import GraphConvolution
import torch.optim as optim
import torch.nn.functional as f

device = 'cuda:0'
dataset = 'cora'
F, A, y, C, train_mask, val_mask, test_mask = load_data(dataset, device)
A = A.float()

N = A.shape[0]
H = 16
D = F.shape[1]

model = GraphConvolution(N, C, H, D, 2, A).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)

loss_window = [0, 0]
loss_inc = 0

for epoch in range(200):
    model.train()
    output = model(F)
    loss = f.nll_loss(torch.log(output[train_mask]+1e-10), y[train_mask])

    opt.zero_grad()
    loss.backward()
    opt.step()

    train_acc = (output[train_mask].argmax(dim=1) ==
            y[train_mask]).sum().item() / y[train_mask].shape[0]
    print('epochs: {}, train loss: {}, train accuracy: {}'.format(
        (epoch+1), loss, train_acc))

    model.eval()
    loss_val = f.nll_loss(torch.log(output[val_mask]), y[val_mask])
    val_acc = (output[val_mask].argmax(dim=1) ==
            y[val_mask]).sum().item() / y[val_mask].shape[0]
    # print(output[train_mask].argmax(dim=1))
    # print(y[train_mask])
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
        y[test_mask]).sum().item() / y[test_mask].shape[0]
print('test accuracy: {}'.format(test_acc))
