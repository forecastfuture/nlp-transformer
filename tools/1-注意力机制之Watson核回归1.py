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
print(f'y_train: {y_train}')

x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)
print(f'n_test: {n_test}')


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)


# print(F'test: {torch.repeat_interleave(y_train.mean(), n_test)}')
# y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)


x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))

attention_weights = nn.functional.softmax(-(x_repeat - x_train) ** 2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)

print(f'attention_weights: {attention_weights.shape}')

plot_kernel_reg(y_hat)

d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
















