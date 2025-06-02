
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)























