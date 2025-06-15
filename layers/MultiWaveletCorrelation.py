import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import math
from functools import partial
from typing import List, Tuple
from layers.utils import get_filter

class MultiWaveletTransform(nn.Cell):
    """
    1D MultiWavelet Transform (self-attention version)
    """

    def __init__(self, ich=1, k=8, alpha=16, c=128, nCZ=1, L=0, base='legendre', attention_dropout=0.1):
        super(MultiWaveletTransform, self).__init__()
        print('MultiWaveletTransform base:', base)
        self.k = k
        self.c = c
        self.L = L
        self.nCZ = nCZ

        self.Lk0 = nn.Dense(ich, c * k)
        self.Lk1 = nn.Dense(c * k, ich)
        self.MWT_CZ = nn.CellList([MWT_CZ1d(k, alpha, L, c, base) for _ in range(nCZ)])

    def construct(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        values = values[:, :L, :, :]  # ensure same length
        values = values.view(B, L, -1)

        V = self.Lk0(values).view(B, L, self.c, -1)
        for i in range(self.nCZ):
            V = self.MWT_CZ[i](V)
            if i < self.nCZ - 1:
                V = nn.ReLU()(V)

        V = self.Lk1(V.view(B, L, -1))
        V = V.view(B, L, H, E)
        return V, None

class FourierCrossAttentionW(nn.Cell):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh', mode_select_method='random'):
        super(FourierCrossAttentionW, self).__init__()
        print('FourierCrossAttentionW used!')
        self.modes1 = modes
        self.activation = activation

    def construct(self, q, k, v, mask=None):
        B, L, E, H = q.shape
        xq = q.transpose(0, 3, 2, 1)  # (B, H, E, L)
        xk = k.transpose(0, 3, 2, 1)
        xv = v.transpose(0, 3, 2, 1)

        xq_ft = ops.fft_rfft(xq)
        xk_ft = ops.fft_rfft(xk)
        xv_ft = ops.fft_rfft(xv)

        index_q = list(range(min(L // 2, self.modes1)))
        index_kv = list(range(min(xv.shape[-1] // 2, self.modes1)))

        xq_ft_ = ops.gather(xq_ft, ms.Tensor(index_q, ms.int32), 3)
        xk_ft_ = ops.gather(xk_ft, ms.Tensor(index_kv, ms.int32), 3)

        xqk_ft = ops.einsum('bhex,bhey->bhxy', xq_ft_, xk_ft_)
        if self.activation == 'tanh':
            xqk_ft = ops.tanh(xqk_ft)
        elif self.activation == 'softmax':
            xqk_ft = ops.softmax(ops.abs(xqk_ft), axis=-1)
        else:
            raise ValueError(f'{self.activation} not implemented')

        xqkv_ft = ops.einsum('bhxy,bhey->bhex', xqk_ft, xk_ft_)
        out_ft = ops.zeros((B, H, E, L // 2 + 1), ms.complex64)

        for i, idx in enumerate(index_q):
            out_ft[:, :, :, idx] = xqkv_ft[:, :, :, i]

        out = ops.fft_irfft(out_ft, n=xq.shape[-1])
        return out.transpose(0, 3, 2, 1), None

class MultiWaveletCross(nn.Cell):
    """
    1D MultiWavelet Cross Attention Layer
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, c=64, k=8, ich=512, L=0, base='legendre', mode_select_method='random', activation='tanh'):
        super(MultiWaveletCross, self).__init__()
        print('MultiWaveletCross base:', base)
        self.c = c
        self.k = k
        self.L = L

        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        self.ec_s = ms.Parameter(ms.Tensor(np.concatenate((H0.T, H1.T), axis=0)), requires_grad=False)
        self.ec_d = ms.Parameter(ms.Tensor(np.concatenate((G0.T, G1.T), axis=0)), requires_grad=False)
        self.rc_e = ms.Parameter(ms.Tensor(np.concatenate((H0 @ PHI0, G0 @ PHI0), axis=0)), requires_grad=False)
        self.rc_o = ms.Parameter(ms.Tensor(np.concatenate((H1 @ PHI1, G1 @ PHI1), axis=0)), requires_grad=False)

        self.Lk = nn.Dense(ich, c * k)
        self.Lq = nn.Dense(ich, c * k)
        self.Lv = nn.Dense(ich, c * k)
        self.out = nn.Dense(c * k, ich)

        self.attn1 = FourierCrossAttentionW(in_channels, out_channels, seq_len_q, seq_len_kv, modes, activation)
        self.attn2 = FourierCrossAttentionW(in_channels, out_channels, seq_len_q, seq_len_kv, modes, activation)
        self.attn3 = FourierCrossAttentionW(in_channels, out_channels, seq_len_q, seq_len_kv, modes, activation)
        self.attn4 = FourierCrossAttentionW(in_channels, out_channels, seq_len_q, seq_len_kv, modes, activation)

    def wavelet_transform(self, x):
        xa = ops.concat([x[:, ::2, :, :], x[:, 1::2, :, :]], axis=-1)
        d = ops.matmul(xa, self.ec_d)
        s = ops.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        assert ich == 2 * self.k
        x_e = ops.matmul(x, self.rc_e)
        x_o = ops.matmul(x, self.rc_o)

        out = ops.zeros((B, N * 2, c, self.k), ms.float32)
        out[:, ::2, :, :] = x_e
        out[:, 1::2, :, :] = x_o
        return out

    def construct(self, q, k, v, mask=None):
        B, N, H, E = q.shape
        q = q.view(B, N, -1)
        k = k.view(B, N, -1)
        v = v.view(B, N, -1)

        q = self.Lq(q).view(B, N, self.c, self.k)
        k = self.Lk(k).view(B, N, self.c, self.k)
        v = self.Lv(v).view(B, N, self.c, self.k)

        ns = int(math.log2(N))
        extra = 2**math.ceil(math.log2(N)) - N
        q = ops.concat([q, q[:, :extra, :, :]], axis=1)
        k = ops.concat([k, k[:, :extra, :, :]], axis=1)
        v = ops.concat([v, v[:, :extra, :, :]], axis=1)

        Ud, Us = [], []

        for _ in range(ns - self.L):
            d_q, q = self.wavelet_transform(q)
            d_k, k = self.wavelet_transform(k)
            d_v, v = self.wavelet_transform(v)

            Ud.append(self.attn1(d_q, d_k, d_v, mask)[0] + self.attn2(q, k, v, mask)[0])
            Us.append(self.attn3(q, k, v, mask)[0])

        v = self.attn4(q, k, v, mask)[0]

        for i in reversed(range(len(Ud))):
            v = v + Us[i]
            v = ops.concat((v, Ud[i]), axis=-1)
            v = self.evenOdd(v)

        v = self.out(v[:, :N, :, :].view(B, N, -1))
        return v, None

class MWT_CZ1d(nn.Cell):
    def __init__(self, k=3, alpha=64, L=0, c=1, base='legendre'):
        super(MWT_CZ1d, self).__init__()
        self.k = k
        self.L = L
        self.A = SparseKernelFT1d(k, alpha, c)
        self.B = SparseKernelFT1d(k, alpha, c)
        self.C = SparseKernelFT1d(k, alpha, c)
        self.T0 = nn.Dense(k, k)

        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        self.ec_s = ms.Parameter(ms.Tensor(np.concatenate((H0.T, H1.T), axis=0)), requires_grad=False)
        self.ec_d = ms.Parameter(ms.Tensor(np.concatenate((G0.T, G1.T), axis=0)), requires_grad=False)
        self.rc_e = ms.Parameter(ms.Tensor(np.concatenate((H0 @ PHI0, G0 @ PHI0), axis=0)), requires_grad=False)
        self.rc_o = ms.Parameter(ms.Tensor(np.concatenate((H1 @ PHI1, G1 @ PHI1), axis=0)), requires_grad=False)

    def wavelet_transform(self, x):
        xa = ops.concat([x[:, ::2, :, :], x[:, 1::2, :, :]], axis=-1)
        d = ops.matmul(xa, self.ec_d)
        s = ops.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape
        x_e = ops.matmul(x, self.rc_e)
        x_o = ops.matmul(x, self.rc_o)
        out = ops.zeros((B, N*2, c, self.k), ms.float32)
        out[:, ::2, :, :] = x_e
        out[:, 1::2, :, :] = x_o
        return out

    def construct(self, x):
        B, N, c, k = x.shape
        ns = int(math.log2(N))
        extra = 2**math.ceil(math.log2(N)) - N
        x = ops.concat([x, x[:, :extra, :, :]], axis=1)

        Ud, Us = [], []

        for _ in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud.append(self.A(d) + self.B(x))
            Us.append(self.C(d))

        x = self.T0(x)
        for i in reversed(range(len(Ud))):
            x = x + Us[i]
            x = ops.concat((x, Ud[i]), axis=-1)
            x = self.evenOdd(x)

        return x[:, :N, :, :]

class SparseKernelFT1d(nn.Cell):
    def __init__(self, k, alpha, c=1):
        super(SparseKernelFT1d, self).__init__()
        self.modes1 = alpha
        self.scale = 1 / (c * k)**2
        self.weights1 = ms.Parameter(self.scale * ops.randn((c*k, c*k, alpha), dtype=ms.complex64))

    def compl_mul1d(self, x, weights):
        return ops.einsum('bix,iox->box', x, weights)

    def construct(self, x):
        B, N, c, k = x.shape
        x = x.view(B, N, -1).transpose(0, 2, 1)
        x_ft = ops.fft_rfft(x)
        l = min(self.modes1, x_ft.shape[-1])
        out_ft = ops.zeros_like(x_ft)
        out_ft[:, :, :l] = self.compl_mul1d(x_ft[:, :, :l], self.weights1[:, :, :l])
        out = ops.fft_irfft(out_ft, n=N)
        return out.transpose(0, 2, 1).view(B, N, c, k)
