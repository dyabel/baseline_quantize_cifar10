from utils import clip,shift
import torch
print(clip(torch.tensor(8.0),2))
print(shift(torch.tensor(0.00001,dtype=float)))
loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.CrossEntropyLoss()
input = torch.autograd.Variable(torch.randn(3,4))
target = torch.autograd.Variable(torch.randn(3,4))
loss = loss_fn(input, target)
print(input); print(target); print(loss)
print(input.size(), target.size(), loss.size())
