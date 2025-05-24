import math
import torch
from torch import nn
from torch.nn import functional as F
# import d2l as dltools
from d2l import torch as d2l


def get_lstm_params(vocab_size, num_hiddens, device):
    """
    lstm 标准模块
    i: 输入门 input gate
    f: 遗忘门 forget gate
    o: 输出门 output gate
    c: 候选记忆 candidate memory cell
    初始化LSTM模型参数"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        """正态分布"""
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        """转换参数"""
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆参数

    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)  # 用于启用张量的自动梯度计算
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    """初始化LSTM状态"""
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    """LSTM模型"""
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state

    outputs = []

    # 前向传播
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)  # 输入门
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)  # 遗忘门
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)  # 输出门
        C_tilde = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)  # 候选记忆

        # 【1：遗忘门 2：上一轮的记忆 3：输入门 4：候选记忆】
        C = F * C + I * C_tilde  # 更新本轮记忆
        # 输出门 + 本轮记忆
        H = O * torch.tanh(C)  # 隐藏层输出
        Y = (H @ W_hq) + b_q   # 输出层输出
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


class LSTMModel:
    """LSTM模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device
        self.params = get_params(vocab_size, num_hiddens, device)   # 初始化参数
        self.init_state = init_state                               # 初始化状态
        self.forward_fn = forward_fn                               # 前向传播

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)              # 前向传播

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens, self.device)  # 初始化状态


def predict_ch8(prefix, num_preds, lstm_model, vocab):
    """
    预测
    prefix: 前缀
    num_preds: 预测的字符数
    lstm_model: 模型
    vocab: 词汇表
    """
    state = lstm_model.begin_state(batch_size=1, device=device)  # 初始化状态
    outputs = [vocab[prefix[0]]]  # 输出
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))  # 获取输入
    # 预热期
    for y in prefix[1:]:
        _, state = lstm_model(get_input(), state)  # 前向传播
        outputs.append(vocab[y])  # 输出
    # 预测
    for _ in range(num_preds):
        y, state = lstm_model(get_input(), state)  # 前向传播
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # 输出
    return ''.join([vocab.idx_to_token[i] for i in outputs])  # 返回预测结果


def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    """Defined in :numref:`sec_minibatches`"""
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]


if __name__ == '__main__':
    # 读取数据
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_fashion_mnist(batch_size, num_steps)  # 加载数据
    # 定义模型
    num_hiddens = 256
    device = d2l.try_gpu()
    lstm_model = LSTMModel(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
    # 训练
    num_epochs, lr = 500, 1
    # trainer_fn, states, hyperparams, data_iter,
    #                feature_dim, num_epochs=2
    train_ch11(lstm_model, train_iter, vocab, lr, num_epochs, device)
    # 预测
    print(predict_ch8('time traveller ', 10, lstm_model, vocab))
    print(predict_ch8('traveller ', 10, lstm_model, vocab))









