import math
import d2l.torch as d2l
from torch import nn
import torch


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失总和,词元数量
    # 取数据
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 第一次训练或使用随机抽样时，需要将state初始化
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 梯度释放
            if isinstance(net, torch.nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            # rnn容易梯度爆炸，使用梯度裁剪
            d2l.grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            d2l.grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


