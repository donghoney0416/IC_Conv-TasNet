import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None

        self.init_hidden()

    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)

    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)


class DepthConv1d(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        # self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.conv2d = nn.Conv2d(input_channel, hidden_channel, (1, 1))

        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv2d = nn.Conv2d(hidden_channel, hidden_channel, (3, kernel), dilation=(1, dilation),
                                 # no dilation for Feature
                                 groups=hidden_channel, padding=(1, self.padding))
        # self.dconv2d = nn.Conv2d(hidden_channel, hidden_channel, (kernel, kernel), dilation=(dilation, dilation),     # dilation for Feature
        #                          groups=hidden_channel, padding=(self.padding, self.padding))

        # self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.res_out = nn.Conv2d(hidden_channel, input_channel, (1, 1))
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            # self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)
            self.skip_out = nn.Conv2d(hidden_channel, input_channel, (1, 1))

    def forward(self, input):  # input : B, C, N, L
        B, C, N, L = input.size()

        # bottleneck layer
        output = self.reg1(self.nonlinearity1(self.conv2d(input)))  # B, C, N, L

        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv2d(output)))  # B, C, N, L

        residual = self.res_out(output)  # B, C N, L
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class DepthConv2d_Attention(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv2d_Attention, self).__init__()

        self.causal = causal
        self.skip = skip

        # Attention
        # self.attention = ECA_2D(kernel=5)

        # self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        self.conv2d = nn.Conv2d(input_channel, hidden_channel, (1, 1))

        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv2d = nn.Conv2d(hidden_channel, hidden_channel, (3, kernel), dilation=(1, dilation),
                                 # no dilation for Feature
                                 groups=hidden_channel, padding=(1, self.padding))
        # self.dconv2d = nn.Conv2d(hidden_channel, hidden_channel, (kernel, kernel), dilation=(dilation, dilation),     # dilation for Feature
        #                          groups=hidden_channel, padding=(self.padding, self.padding))

        # self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.res_out = nn.Conv2d(hidden_channel, input_channel, (1, 1))
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            # self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)
            self.skip_out = nn.Conv2d(hidden_channel, input_channel, (1, 1))

    def forward(self, input):  # input : B, C, N, L
        B, C, N, L = input.size()
        # Attention
        # output = self.attention(input)

        # bottleneck layer
        output = self.reg1(self.nonlinearity1(self.conv2d(input)))  # B, C, N, L

        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv2d(output)))  # B, C, N, L

        residual = self.res_out(output)  # B, C N, L
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    def __init__(self, mic_num, ch_dim, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True,
                 causal=False, dilated=True):
        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN_C = nn.Conv2d(mic_num, ch_dim, (1, 1))
        self.BN_N = nn.Conv2d(input_dim, BN_dim, (1, 1))

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    # self.TCN.append(DepthConv1d(ch_dim, ch_dim * 4, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal))
                    self.TCN.append(
                        DepthConv2d_Attention(ch_dim, ch_dim * 4, kernel, dilation=2 ** i, padding=2 ** i, skip=skip,
                                              causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(ch_dim, ch_dim * 4, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # output layer

        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(BN_dim, output_dim, (1, 1)))
        self.output_act = nn.PReLU()
        self.output_C = nn.Conv2d(ch_dim, 1, (1, 1))
        # self.output_N = nn.Conv2d(BN_dim, output_dim, (1, 1))

        self.skip = skip

    def forward(self, input):

        # input shape: (B, C, N, L)

        # normalization
        B, C, N, L = input.size()

        norm = self.LN(input.view(B * C, N, L)).view(B, C, N, L)
        output = self.BN_C(self.BN_N(norm.transpose(1, 2)).transpose(1, 2))

        # output = self.C_attention(output)

        # output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            # output = self.output(skip_connection)
            output = self.output_C(self.output(skip_connection.transpose(1, 2)).transpose(1, 2))

        else:
            output = self.output(output)

        return output