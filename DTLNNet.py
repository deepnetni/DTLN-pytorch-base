import os
import numpy as np
import torch
import torch.nn as nn


class STFT(nn.Module):
    def __init__(self, n_frame, n_hop):
        super(STFT, self).__init__()
        self.n_frame = n_frame
        self.n_hop = n_hop
        self.n_overlap = n_frame - n_hop
        self.eps = torch.finfo(torch.float32).eps
        self.register_buffer("win", torch.from_numpy(self.gen_window()))

    def gen_window(self):
        if self.n_hop * 2 > self.n_frame:
            n_hann = self.n_overlap * 2 - 1
            # matlab start index from 1, while python start from 0
            hann = np.sqrt(np.hanning(n_hann+2))[1:-1]
            win = np.concatenate((
                np.zeros(1),
                hann[:self.n_overlap-1],
                np.ones(self.n_frame - 2 * self.n_overlap - 1),
                hann[self.n_overlap-1:]
            ))
        else:
            n_hann = 2 * self.n_hop - 1
            hann = np.sqrt(np.hanning(n_hann+2))[1:-1]
            win = np.concatenate((
                np.zeros(int(self.n_frame/2) - self.n_hop + 1),
                hann,
                np.zeros(int(self.n_frame/2) - self.n_hop)
            ))

        return win

    def forward(self, x):
        # the shape of y is (N, F, T)
        y = torch.stft(x, n_fft=self.n_frame, hop_length=self.n_hop,
                       window=self.win,
                       pad_mode='constant',
                       return_complex=True,
                       center=True)
        r, i = y.real, y.imag
        mag = torch.absolute(y).float()
        phi = torch.atan2(i + self.eps, r + self.eps).float()
        return mag, phi

class InstanceNorm(nn.Module):
    def __init__(self, n_feats):
        super(InstanceNorm, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.register_parameter('gamma', nn.Parameter(torch.ones(1, 1, n_feats)), requires_grad=True)
        self.register_parameter('beta', nn.Parameter(torch.zeros(1, 1, n_feats)), requires_grad=True)

    def forward(self, x):
        # x of shape (N, T, F), calculate mean and variance
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean(torch.square(x - mean), dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        out = ((x - mean) / var) * self.gamma + self.beta
        return out

class Seperation(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
        super(Seperation, self).__init__()
        # input_size is the lenght of features each frame
        self.lstm_1 = nn.LSTM(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              batch_first=True,
                              dropout=0.0,
                              bidirectional=False)
        self.lstm_2 = nn.LSTM(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              batch_first=True,
                              dropout=0.0,
                              bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, (h, c) = self.lstm_1(x)
        x = self.dropout(x)
        x, _ = self.lstm_2(x)
        x = self.dropout(x)

        mask = self.sigmoid(self.dense(x))
        return mask

class OLA(nn.Module):
    def __init__(self, n_fft, n_hop):
        super(OLA, self).__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.n_overlap = n_fft - n_hop

    def forward(self, x):
        n_batch, n_frame, L = x.shape
        out = torch.zeros((n_batch, (n_frame-1)*self.n_hop+L)).cuda()

        for i in range(n_frame):
            out[..., i*self.n_hop:i*self.n_hop+L] += x[..., i, :]

        return out[..., self.n_overlap:-self.n_overlap]

    def forward2(self, x):
        for btx, batch_data in enumerate(x):
            for idx, frame in enumerate(batch_data):
                if idx == 0:
                    out = frame
                else:
                    out = torch.concat((out[:-self.n_overlap],
                                        out[-self.n_overlap:]+frame[:self.n_overlap],
                                        frame[self.n_overlap:]), dim=0)
            if btx == 0:
                out_ = torch.unsqueeze(out, dim=0)
            else:
                out_ = torch.concat((out_, torch.unsqueeze(out, dim=0)), dim=0)

        return out_[..., self.n_overlap:-self.n_overlap]


class DTLNNet(nn.Module):
    def __init__(self, n_fft, n_hop):
        super(DTLNNet, self).__init__()
        features_each_frame = int(n_fft / 2) + 1
        hidden_size = 128
        dropout = 0.25
        codec_size = 256

        self.stft = STFT(n_fft, n_hop)
        self.stage_1 = Seperation(input_size=features_each_frame,
                                  hidden_size=hidden_size, dropout=dropout)
        # TODO check in_channels
        self.encode = nn.Conv1d(in_channels=n_fft, out_channels=codec_size,
                                kernel_size=1, stride=1, bias=False)
        self.norm = nn.InstanceNorm1d(num_features=codec_size, affine=False)
        self.stage_2 = Seperation(input_size=codec_size,
                                  hidden_size=hidden_size, dropout=dropout)
        self.decode = nn.Conv1d(in_channels=codec_size, out_channels=n_fft,
                                kernel_size=1, stride=1, bias=False)
        self.ola = OLA(n_fft, n_hop)

    def forward(self, x):
        mag, phi = self.stft(x)
        # transfer to (N, Frames, L)
        mag = mag.permute(0, 2, 1)
        phi = phi.permute(0, 2, 1)

        mask = self.stage_1(mag)
        est_stft = mask * mag
        est = est_stft * torch.exp(1j * phi)
        est_t = torch.fft.irfft(est, dim=-1)
        # N, L, F
        x = est_t.permute(0, 2, 1)

        encode = self.encode(x)
        encode = encode.permute(0, 2, 1)
        encode_norm = self.norm(encode)

        mask_2 = self.stage_2(encode_norm)
        est_2 = mask_2 * encode
        x = est_2.permute(0, 2, 1)
        decode = self.decode(x)
        decode = decode.permute(0, 2, 1)

        # OLA
        decode = decode * self.stft.win
        sig = self.ola(decode)

        return sig
