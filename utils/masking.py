import mindspore as ms
import mindspore.ops as ops
import numpy as np
import math

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = (B, 1, L, L)
        ones = ops.ones(mask_shape, dtype=ms.bool_)
        self._mask = ops.triu(ones, diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores):
        # 创建上三角 mask [L, L]
        ones = ops.ones((L, scores.shape[-1]), dtype=ms.bool_)
        _mask = ops.triu(ones, diagonal=1)  # bool tensor

        # broadcast 到 [B, H, L, L]
        _mask_ex = ops.expand_dims(_mask, 0)
        _mask_ex = ops.broadcast_to(_mask_ex, (B, H, L, scores.shape[-1]))

        # 构造索引
        B_idx = ops.arange(B, dtype=ms.int32).reshape((B, 1, 1))
        H_idx = ops.arange(H, dtype=ms.int32).reshape((1, H, 1))

        B_idx = ops.broadcast_to(B_idx, index.shape)
        H_idx = ops.broadcast_to(H_idx, index.shape)

        # [B, H, L, L] → [B, H, u, L]
        indicator = _mask_ex[B_idx, H_idx, index]

        # reshape 回 scores 形状
        self._mask = ops.reshape(indicator, scores.shape)

    @property
    def mask(self):
        return self._mask



class LocalMask():
    def __init__(self, B, L, S, device="cpu"):
        mask_shape = (B, 1, L, S)
        ones = ops.ones(mask_shape, dtype=ms.bool_)
        self.len = math.ceil(np.log2(L))
        mask1 = ops.triu(ones, diagonal=1)
        mask2 = ops.logical_not(ops.triu(ones, diagonal=-self.len))
        self._mask = ops.logical_or(mask1, mask2)

    @property
    def mask(self):
        return self._mask
