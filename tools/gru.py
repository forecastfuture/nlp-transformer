import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


def get_params(vocab_size, num_hiddens, device):
    """
    初始化参数：
    r：Reset Gate（重置门）
    z：Update Gate（更新门）
    h：Hidden State（隐藏状态）
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """正态分布"""
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        """初始化参数 w, b"""
        return (normal((num_inputs, num_hiddens)),
                normal((num_inputs, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()    # 更新门参数
    W_xr, W_hr, b_r = three()    # 重置门参数
    W_xh, W_hh, b_h = three()    # 候选隐藏状态参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    # 梯度设置
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    """初始化GRU状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    """GRU模型"""
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)   # 更新门
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)   # 重置门
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)  # 候选隐藏状态
        H = Z * H + (1 - Z) * H_tilda   # 隐藏状态
        Y = H @ W_hq + b_q   # 输出
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class GRUModel:
    """GRU模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)   # 初始化参数
        self.init_state = init_state                               # 初始化状态
        self.forward_fn = forward_fn                               # 前向传播

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)              # 前向传播

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)  # 初始化状态


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

    # 单独实现 方式一
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_epochs, lr = 500, 1
    model = GRUModel(vocab_size, num_hiddens, device, get_params, init_gru_state, gru)
    # 训练
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    # 简洁写法  方式二
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)










