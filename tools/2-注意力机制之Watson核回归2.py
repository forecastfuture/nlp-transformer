
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


n_train = 50
# print(F'torch.randn(n_train): {torch.randn(n_train)}')
x_train, _ = torch.sort(torch.rand(n_train) * 5)
# print(f"x_train: {x_train}")
#
# print(f'torch.normal(0.0, 0.5, (n_train,)): {torch.normal(0.0, 0.5, (n_train,))}')
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))

X = torch.ones((2, 1, 4))
Y = torch.ones((2, 4, 6))
print(torch.bmm(X, Y).shape)

weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape(2, 10)
print(F'bmm: {torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)).shape}')


class NWKernelRegression(nn.Module):
    def __int__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1, ), requires_grad=True))

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1
        )
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


x_tile = x_train.repeat((n_train, 1))
y_tile = y_train.repeat((n_train, 1))
keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))


# 执行训练
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))














