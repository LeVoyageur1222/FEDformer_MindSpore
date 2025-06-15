import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = np.random.permutation(seq_len // 2)[:modes]
    else:
        index = np.arange(modes)
    index = np.sort(index)
    return index.tolist()


def complex_mul(a_real, a_imag, b_real, b_imag, mode):
    """使用实数表示执行复数乘法"""
    einsum = ops.Einsum(mode)
    real = einsum((a_real, b_real)) - einsum((a_imag, b_imag))
    imag = einsum((a_real, b_imag)) + einsum((a_imag, b_real))
    return real, imag  # 返回实部和虚部的实数张量


class FourierBlock(nn.Cell):
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('FourierBlock used!')
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        self.seq_len = seq_len
        self.scale = (1 / (in_channels * out_channels))**0.5

        real = self.scale * ops.randn((8, in_channels // 8, out_channels // 8, len(self.index)), dtype=ms.float32)
        imag = self.scale * ops.randn((8, in_channels // 8, out_channels // 8, len(self.index)), dtype=ms.float32)
        
        # 只存储实数权重
        self.weights1_real = ms.Parameter(self.scale * ops.randn((8, in_channels//8, out_channels//8, len(self.index))))
        self.weights1_imag = ms.Parameter(self.scale * ops.randn((8, in_channels//8, out_channels//8, len(self.index))))
        
        self.rfft = ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        self.irfft = ops.FFTWithSize(signal_ndim=1, inverse=True, real=True, signal_sizes=(seq_len,))

    def construct(self, q, k, v, mask=None):
        B, L, H, E = q.shape
        x = ops.transpose(q, (0, 2, 3, 1)).reshape((-1, L))
        
        # FFT 并分离实部和虚部
        x_ft_complex = self.rfft(x).reshape((B, H, E, -1))
        x_ft_real = ops.real(x_ft_complex)
        x_ft_imag = ops.imag(x_ft_complex)
        
        # 初始化输出
        out_ft_real = ops.zeros((B, H, E, x_ft_real.shape[-1]), dtype=ms.float32)
        out_ft_imag = ops.zeros((B, H, E, x_ft_imag.shape[-1]), dtype=ms.float32)
        
        # 使用实数表示执行复数乘法
        for wi, i in enumerate(self.index):
            a_real = x_ft_real[:, :, :, i]
            a_imag = x_ft_imag[:, :, :, i]
            b_real = self.weights1_real[:, :, :, wi]
            b_imag = self.weights1_imag[:, :, :, wi]
            
            real_part, imag_part = complex_mul(a_real, a_imag, b_real, b_imag, 'bhi,hio->bho')
            
            out_ft_real[:, :, :, i] = real_part
            out_ft_imag[:, :, :, i] = imag_part
        
        # 组合为复数进行逆FFT
        out_ft = ops.Complex()(out_ft_real, out_ft_imag)
        out = self.irfft(out_ft.reshape((-1, out_ft.shape[-1]))).reshape((B, H, E, L))
        out = ops.transpose(out, (0, 3, 1, 2))
        return out, None


class FourierCrossAttention(nn.Cell):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64, mode_select_method='random',
                 activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print('FourierCrossAttention used!')
        self.index_q = get_frequency_modes(seq_len_q, modes, mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes, mode_select_method)
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.scale = (1 / (in_channels * out_channels))**0.5

        real = self.scale * ops.randn((8, in_channels // 8, out_channels // 8, len(self.index_q)), dtype=ms.float32)
        imag = self.scale * ops.randn((8, in_channels // 8, out_channels // 8, len(self.index_q)), dtype=ms.float32)
        
        # 只存储实数权重
        self.weights1_real = ms.Parameter(self.scale * ops.randn((8, in_channels//8, out_channels//8, len(self.index_q))))
        self.weights1_imag = ms.Parameter(self.scale * ops.randn((8, in_channels//8, out_channels//8, len(self.index_q))))
        
        self.rfft_q = ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        self.rfft_kv = ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        self.irfft = ops.FFTWithSize(signal_ndim=1, inverse=True, real=True, signal_sizes=(seq_len_q,))

    def construct(self, q, k, v, mask=None):
        B, L, H, E = q.shape
        _, L_k, _, _ = k.shape
        
        # 准备输入
        xq = ops.transpose(q, (0, 2, 3, 1)).reshape((-1, L))
        xk = ops.transpose(k, (0, 2, 3, 1)).reshape((-1, L_k))
        
        # FFT 并分离实部和虚部
        xq_ft = self.rfft_q(xq).reshape((B, H, E, -1))
        xk_ft = self.rfft_kv(xk).reshape((B, H, E, -1))
        
        xq_ft_real = ops.real(xq_ft)
        xq_ft_imag = ops.imag(xq_ft)
        xk_ft_real = ops.real(xk_ft)
        xk_ft_imag = ops.imag(xk_ft)
        
        # 选择频率分量
        xq_ft_real_select = ops.gather(xq_ft_real, ms.Tensor(self.index_q, ms.int32), 3)
        xq_ft_imag_select = ops.gather(xq_ft_imag, ms.Tensor(self.index_q, ms.int32), 3)
        xk_ft_real_select = ops.gather(xk_ft_real, ms.Tensor(self.index_kv, ms.int32), 3)
        xk_ft_imag_select = ops.gather(xk_ft_imag, ms.Tensor(self.index_kv, ms.int32), 3)
        
        # 复数乘法：xq * conj(xk)
        xqk_real, xqk_imag = complex_mul(
            xq_ft_real_select, xq_ft_imag_select,
            xk_ft_real_select, -xk_ft_imag_select,  # 取共轭
            'bhex,bhey->bhxy'
        )
        
        # 应用激活函数（使用实数表示）
        if self.activation == 'tanh':
            xqk_real = ops.tanh(xqk_real)
            xqk_imag = ops.tanh(xqk_imag)
        elif self.activation == 'softmax':
            magnitude = ops.sqrt(xqk_real**2 + xqk_imag**2 + 1e-8)
            softmax = ops.softmax(magnitude, axis=-1)
            # 保持相位不变
            phase = ops.atan2(xqk_imag, xqk_real)
            xqk_real = softmax * ops.cos(phase)
            xqk_imag = softmax * ops.sin(phase)
        
        # 再次复数乘法：(xqk) * xk
        xqkv_real, xqkv_imag = complex_mul(
            xqk_real, xqk_imag,
            xk_ft_real_select, xk_ft_imag_select,
            'bhxy,bhey->bhex'
        )
        
        # 最后与权重相乘
        xqkvw_real, xqkvw_imag = complex_mul(
            xqkv_real, xqkv_imag,
            self.weights1_real, self.weights1_imag,
            'bhex,heox->bhox'
        )
        
        # 准备逆FFT输入
        out_ft_real = ops.zeros((B, H, E, L//2+1), dtype=ms.float32)
        out_ft_imag = ops.zeros((B, H, E, L//2+1), dtype=ms.float32)
        
        for wi, i in enumerate(self.index_q):
            out_ft_real[:, :, :, i] = xqkvw_real[:, :, :, wi]
            out_ft_imag[:, :, :, i] = xqkvw_imag[:, :, :, wi]
        
        # 组合为复数进行逆FFT
        out_ft = ops.Complex()(out_ft_real, out_ft_imag)
        out = self.irfft(out_ft.reshape((-1, out_ft.shape[-1]))).reshape((B, H, E, L))
        out = ops.transpose(out, (0, 3, 1, 2))
        return out, None