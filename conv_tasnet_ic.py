import torch
from torch.autograd import Variable

from config import config

from utility import models_attention_set


class TasNet(torch.nn.Module):
    def __init__(self):
        super(TasNet, self).__init__()
        # hyper parameters
        self.mic_num = config.model_mic_num
        self.num_spk = config.model_num_spk

        # increased enc dim
        self.enc_dim = config.model_enc_dim
        self.feature_dim = config.model_feature_dim

        self.ch_dim = config.model_ch_dim

        self.win = int(config.sample_rate * config.model_win / 1000)
        self.stride = self.win // 2

        self.layer = config.model_layer
        self.stack = config.model_stack
        self.kernel = config.model_kernel

        self.causal = config.model_causal

        # input encoder
        self.encoder = torch.nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.stride)

        # TCN separator
        self.TCN = models_attention_set.TCN(self.mic_num, self.ch_dim, self.enc_dim, self.enc_dim * self.num_spk,
                                            self.feature_dim, self.feature_dim * 4,  # single modified
                                            self.layer, self.stack, self.kernel, causal=self.causal)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = torch.nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, input):
        # Padding
        output, rest = self.pad_signal(input)

        batch_size = output.size(0)
        num_ch = output.size(1)
        enc_output = self.encoder(output.view(batch_size * num_ch, 1, -1)).view(batch_size, num_ch, self.enc_dim, -1)  # B, C, N, L

        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L

        # reference mic = mic5
        masked_output = enc_output[:, config.reference_channel_idx:config.reference_channel_idx+1] * masks

        # waveform decoder
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L

        output = output.view(batch_size, self.num_spk, -1)  # B, C, T

        return output
