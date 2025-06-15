import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import math

import numpy as np

class AutoCorrelation(nn.Cell):
    """
    AutoCorrelation Mechanism:
    - (1) period-based dependencies discovery
    - (2) time delay aggregation
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        print('AutoCorrelation used!')
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=1.0 - attention_dropout)
        
        # 使用FFTWithSize替代原始FFT函数
        self.rfft = ops.FFTWithSize(signal_ndim=1, inverse=False, real=True)
        self.irfft = ops.FFTWithSize(signal_ndim=1, inverse=True, real=True)
        self.cast = ops.Cast()

    def time_delay_agg_training(self, values, corr):
        """时间延迟聚合 - 训练模式"""
        B, H, D, L = values.shape
        top_k = int(self.factor * math.log(L))

        # 确保输入为实数
        corr = self.cast(corr, ms.float32)
        
        mean_value = corr.mean(axis=1).mean(axis=1)  # [B, L]
        index = ops.topk(mean_value.mean(axis=0), top_k)[1]  # [top_k]
        weights = ops.stack([mean_value[:, index[i]] for i in range(top_k)], axis=-1)  # [B, top_k]
        tmp_corr = ops.Softmax(axis=-1)(weights)  # [B, top_k]

        delays_agg = ops.zeros_like(values)
        for i in range(top_k):
            shifted = ops.roll(values, shifts=-int(index[i].asnumpy()), dims=-1)
            weight_expand = tmp_corr[:, i].view(B, 1, 1, 1).broadcast_to(values.shape)
            delays_agg += shifted * weight_expand

        return delays_agg

#     def time_delay_agg_inference(self, values, corr):
#         """时间延迟聚合 - 推理模式"""
#         B, H, D, L = values.shape
#         top_k = int(self.factor * math.log(L))

#         # 确保输入为实数
#         corr = self.cast(corr, ms.float32)
        
#         mean_value = corr.mean(axis=1).mean(axis=1)  # [B, L]
#         weights = ops.topk(mean_value, top_k)[0]  # [B, top_k]
#         delay = ops.topk(mean_value, top_k)[1]   # [B, top_k]
#         tmp_corr = ops.Softmax(axis=-1)(weights)

#         delays_agg = ops.zeros_like(values)
#         values_pad = ops.concat([values, values], axis=-1)  # pad to 2L for easy rolling

#         for i in range(top_k):
#             roll_index = ops.arange(L).view(1, 1, 1, L) + delay[:, i].view(B, 1, 1, 1)
#             roll_index = roll_index % (2 * L)
#             pattern = ops.gather_elements(values_pad, dim=-1, index=roll_index)
#             weight_expand = tmp_corr[:, i].view(B, 1, 1, 1).broadcast_to(values.shape)
#             delays_agg += pattern * weight_expand

#         return delays_agg

    def time_delay_agg_inference(self, values, corr):
        B, H, D, L = values.shape
        top_k = int(self.factor * math.log(L))

        mean_value = corr.mean(axis=1).mean(axis=1)  # [B, L]
        weights = ops.topk(mean_value, top_k)[0]  # [B, top_k]
        delay = ops.topk(mean_value, top_k)[1].astype(ms.int32)  # [B, top_k]
        tmp_corr = ops.Softmax(axis=-1)(weights)

        delays_agg = ops.zeros_like(values)
        values_pad = ops.concat([values, values], axis=-1)  # pad to 2L for easy rolling

        roll_base = ms.Tensor(np.arange(L), dtype=ms.int32).view(1, 1, 1, L)

        for i in range(top_k):
            shift = delay[:, i].view(B, 1, 1, 1)
            roll_index = (roll_base + shift) % (2 * L)  # [B, 1, 1, L]
            roll_index = roll_index.broadcast_to((B, H, D, L))  # 明确 shape

            pattern = ops.gather_elements(values_pad, -1, roll_index)
            weight_expand = tmp_corr[:, i].view(B, 1, 1, 1).broadcast_to(values.shape)
            delays_agg += pattern * weight_expand

        return delays_agg

    def complex_mul(self, a_real, a_imag, b_real, b_imag, conj=False):
        """复数乘法（分解为实部和虚部）"""
        if conj:
            # (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
            real_part = a_real * b_real + a_imag * b_imag
            imag_part = a_imag * b_real - a_real * b_imag
        else:
            # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            real_part = a_real * b_real - a_imag * b_imag
            imag_part = a_real * b_imag + a_imag * b_real
        return real_part, imag_part

    def construct(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # 确保输入为float32
        queries = self.cast(queries, ms.float32)
        keys = self.cast(keys, ms.float32)
        values = self.cast(values, ms.float32)
        
        if L > S:
            zeros = ops.zeros((B, L - S, H, E), ms.float32)
            values = ops.concat([values, zeros], axis=1)
            keys = ops.concat([keys, zeros], axis=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        queries = ops.transpose(queries, (0, 2, 3, 1))  # (B, H, E, L)
        keys = ops.transpose(keys, (0, 2, 3, 1))         # (B, H, E, L)
        
        # FFT变换 - 分离实部和虚部
        q_fft_complex = self.rfft(queries)
        k_fft_complex = self.rfft(keys)
        
        # 分解为实部和虚部
        q_fft_real = ops.real(q_fft_complex)
        q_fft_imag = ops.imag(q_fft_complex)
        k_fft_real = ops.real(k_fft_complex)
        k_fft_imag = ops.imag(k_fft_complex)
        
        # 复数乘法: q * conj(k) - 使用自定义复数乘法
        res_real, res_imag = self.complex_mul(
            q_fft_real, q_fft_imag, 
            k_fft_real, k_fft_imag,
            conj=True
        )
        
        # 组合为复数进行逆FFT
        res_complex = ops.Complex()(res_real, res_imag)
        
        # 逆FFT变换 - 指定输出长度
        self.irfft = ops.FFTWithSize(signal_ndim=1, inverse=True, real=True, signal_sizes=(L,))
        corr = self.irfft(res_complex)
        
        # 确保自相关结果为实数
        corr = self.cast(corr, ms.float32)

        if self.training:
            V = self.time_delay_agg_training(ops.transpose(values, (0, 2, 3, 1)), corr)
        else:
            V = self.time_delay_agg_inference(ops.transpose(values, (0, 2, 3, 1)), corr)

        V = ops.transpose(V, (0, 3, 1, 2))  # (B, L, H, E)

        if self.output_attention:
            return V, ops.transpose(corr, (0, 3, 1, 2))
        else:
            return V, None

class AutoCorrelationLayer(nn.Cell):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.cast = ops.Cast()

    def construct(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 确保输入为float32
        queries = self.cast(queries, ms.float32)
        keys = self.cast(keys, ms.float32)
        values = self.cast(values, ms.float32)

        queries = ops.reshape(self.query_projection(queries), (B, L, H, -1))
        keys = ops.reshape(self.key_projection(keys), (B, S, H, -1))
        values = ops.reshape(self.value_projection(values), (B, S, H, -1))

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = ops.reshape(out, (B, L, -1))

        return self.out_projection(out), attn