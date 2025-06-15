import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import math

from utils.masking import ProbMask

class FullAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(p=1.0 - attention_dropout)  # ms定义是keep_prob
        self.softmax = nn.Softmax(axis=-1)
        self.einsum = ops.Einsum('blhe,bshe->bhls')
        self.einsum_v = ops.Einsum('bhls,bshd->blhd')

    def construct(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = self.einsum((queries, keys))  # [B, H, L, S]

        if self.mask_flag and attn_mask is not None:
            scores = ops.masked_fill(scores, attn_mask.mask, -1e9)  # 避免数值爆炸

        A = self.dropout(self.softmax(scale * scores))
        V = self.einsum_v((A, values))
        # V = ops.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V, A
        else:
            return V, None

class ProbAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        # MindSpore dropout使用保留概率，所以用1减去
        self.dropout = nn.Dropout(p=1.0 - attention_dropout)
        self.softmax = nn.Softmax(axis=-1)
        
    def _batch_gather_3d(self, x, index):
        """
        x: Tensor [B, H, L, D]
        index: Tensor [B, H, S] — 索引值沿 axis=2
        return: [B, H, S, D]
        """
        B, H, L, D = x.shape
        S = index.shape[-1]

        # 展平索引构建 gather 索引: [B*H*S, 3]
        b_idx = ops.arange(B, dtype=ms.int32).reshape((B, 1, 1))
        b_idx = ops.tile(b_idx, (1, H, S))

        h_idx = ops.arange(H, dtype=ms.int32).reshape((1, H, 1))
        h_idx = ops.tile(h_idx, (B, 1, S))

        idx_flat = ops.stack([b_idx, h_idx, index.astype(ms.int32)], axis=-1).reshape((-1, 3))

        # 对 x 进行 flatten
        x_flat = x.reshape((B * H * L, D))
        gather_index = idx_flat[:, 0] * H * L + idx_flat[:, 1] * L + idx_flat[:, 2]  # shape [B*H*S]
        out = ops.gather(x_flat, gather_index, axis=0)  # shape [B*H*S, D]
        return out.reshape((B, H, S, D))

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Args:
            Q: Tensor, shape [B, H, L_Q, D]
            K: Tensor, shape [B, H, L_K, D]
            sample_k: int, number of key positions to sample
            n_top: int, number of top-k queries to keep
        Returns:
            Q_K: Tensor, [B, H, n_top, L_K]
            M_top: Tensor, [B, H, n_top]
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # === 1. 随机采样 sample_k 个 key 对于每个 query ===
        index_sample_np = np.random.randint(0, L_K, size=(L_Q, sample_k))
        K_sample_list = []

        for i in range(L_Q):
            idx = Tensor(index_sample_np[i], ms.int32)  # [sample_k]
            gathered = ops.gather(K, idx, axis=2)  # [B, H, sample_k, D]
            K_sample_list.append(gathered.expand_dims(2))  # → [B, H, 1, sample_k, D]

        # 拼接 → [B, H, L_Q, sample_k, D]
        K_sample = ops.concat(K_sample_list, axis=2)
        
        # print(f"K_sample shape is {K_sample.shape}") # K_sample shape is (32, 8, 96, 15, 64)

        # === 2. Q × K_sample^T → [B, H, L_Q, sample_k]
        Q_exp = ops.expand_dims(Q, -2)  # [B, H, L_Q, 1, D]
        K_sample_T = ops.transpose(K_sample, (0, 1, 2, 4, 3))  # [B, H, L_Q, D, sample_k]
        Q_K_sample = ops.matmul(Q_exp, K_sample_T).squeeze(-2)  # [B, H, L_Q, sample_k]

        # === 3. 稀疏度评分并选 Top-k Q ===
        Q_K_max = ops.max(Q_K_sample, axis=-1)[0]  # [B, H, L_Q]
        Q_K_mean = ops.mean(Q_K_sample, axis=-1)   # [B, H, L_Q]
        M = Q_K_max - Q_K_mean                     # [B, H, L_Q]
        # print(f"Q_K_max shape is {Q_K_max.shape}, Q_K_mean shape is {Q_K_mean.shape}, M shape is {M.shape}")

        M_top = ops.top_k(M, n_top, sorted=False)[1]  # [B, H, n_top]
        # print(f"M_top shape is {M_top.shape}")

        # === 4. 收集 top-k Q_reduce ===
        # Q_reduce = ops.gather(Q, M_top, axis=2)  # [B, H, n_top, D]
        Q_reduce = self._batch_gather_3d(Q, M_top)
        # print(f"Q_reduce shape is {Q_reduce.shape}")

        # === 5. Q_reduce × K^T → Q_K ===
        K_T = ops.transpose(K, (0, 1, 3, 2))  # [B, H, D, L_K]
        Q_K = ops.matmul(Q_reduce, K_T)      # [B, H, n_top, L_K]

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_mean = V.mean(axis=-2)  # [B, H, D]
            context = V_mean.reshape(B, H, 1, D).broadcast_to((B, H, L_Q, D))
        else:
            context = V.cumsum(axis=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        u = index.shape[2]  # top-u的数量

        # print(f"scores shape is {scores.shape}")
        
        # 应用注意力掩码
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)
            scores = ops.masked_fill(scores, attn_mask.mask, -1e9)
        
        # 计算注意力权重 [B, H, u, L_V]
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        # print(f"attn shape is {attn.shape}")
        
        
        # 计算更新向量 [B, H, u, D]
        update = ops.matmul(attn, V)
        
        # 构建scatter索引 [B*H*u, 3]
        b_idx = ops.arange(B, dtype=ms.int32).view(B, 1, 1).broadcast_to((B, H, u))
        h_idx = ops.arange(H, dtype=ms.int32).view(1, H, 1).broadcast_to((B, H, u))
        index = index.astype(ms.int32)
        scatter_idx = ops.stack([b_idx, h_idx, index], axis=-1).reshape(-1, 3)
        
        # 重塑更新向量 [B*H*u, D]
        update = update.reshape(-1, D)
        
        # 使用scatter更新上下文
        context_out = ops.tensor_scatter_update(context_in, scatter_idx, update)
        
        # 如果需要输出注意力矩阵
        if self.output_attention:
            attns = ops.ones((B, H, L_V, L_V), ms.float32) / L_V
            attns = ops.tensor_scatter_update(
                attns, 
                scatter_idx, 
                attn.reshape(-1, L_V)
            )
            return context_out, attns
        
        return context_out, None

    def construct(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # 调整维度顺序 [B, L, H, D] -> [B, H, L, D]
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # 计算采样数量
        U_part = min(self.factor * math.ceil(math.log(L_K)), L_K)
        u = min(self.factor * math.ceil(math.log(L_Q)), L_Q)

        # 获取top-k注意力分数
        scores_top, index = self._prob_QK(queries, keys, U_part, u)
        
        # print(f"scores_top shape is {scores_top.shape}")

        # 应用缩放因子
        scale = self.scale or 1.0 / math.sqrt(D)
        scores_top = scores_top * scale
        
        # print(f"scores_top shape after scale is {scores_top.shape}")

        # 初始化并更新上下文
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        # 恢复维度顺序 [B, H, L, D] -> [B, L, H, D]
        return context.transpose(0, 2, 1, 3), attn

class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = ops.reshape(self.query_projection(queries), (B, L, H, -1))
        keys = ops.reshape(self.key_projection(keys), (B, S, H, -1))
        values = ops.reshape(self.value_projection(values), (B, S, H, -1))

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = ops.reshape(out, (B, L, -1))

        return self.out_projection(out), attn
