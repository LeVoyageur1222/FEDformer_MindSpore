import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import math

class MyLayerNorm(nn.Cell):
    """
    Special designed LayerNorm for seasonal part (去除均值)
    """
    def __init__(self, channels):
        super(MyLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm((channels,))

    def construct(self, x):
        x_hat = self.layernorm(x)
        bias = x_hat.mean(axis=1, keep_dims=True)
        return x_hat - bias

class MovingAvg(nn.Cell):
    """
    Moving average block to highlight trend
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, pad_mode='valid')

    def construct(self, x):
        # padding manually at both ends
        front = ops.tile(x[:, :1, :], (1, self.kernel_size - 1 - self.kernel_size // 2, 1))
        end = ops.tile(x[:, -1:, :], (1, self.kernel_size // 2, 1))
        x = ops.concat((front, x, end), axis=1)
        x = ops.transpose(x, (0, 2, 1))  # (B, C, L)
        x = self.avg(x)
        x = ops.transpose(x, (0, 2, 1))  # (B, L, C)
        return x

class SeriesDecomp(nn.Cell):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def construct(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class SeriesDecompMulti(nn.Cell):
    """
    Multi-scale Series decomposition block
    """
    def __init__(self, kernel_size_list):
        super(SeriesDecompMulti, self).__init__()
        self.moving_avg_list = nn.CellList([MovingAvg(kernel, stride=1) for kernel in kernel_size_list])
        self.linear = nn.Dense(1, len(kernel_size_list))

    def construct(self, x):
        moving_mean = []
        for avg_layer in self.moving_avg_list:
            moving_mean.append(avg_layer(x).expand_dims(-1))  # [B, L, C, 1]
        moving_mean = ops.concat(moving_mean, axis=-1)  # [B, L, C, num_scales]

        weights = nn.Softmax(axis=-1)(self.linear(x.expand_dims(-1)))  # [B, L, C, num_scales]
        moving_mean_weighted = (moving_mean * weights).sum(axis=-1)  # [B, L, C]

        res = x - moving_mean_weighted
        return res, moving_mean_weighted

class EncoderLayer(nn.Cell):
    """
    Autoformer encoder layer with progressive decomposition
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, has_bias=True)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, has_bias=True)

        if isinstance(moving_avg, list):
            self.decomp1 = SeriesDecompMulti(moving_avg)
            self.decomp2 = SeriesDecompMulti(moving_avg)
        else:
            self.decomp1 = SeriesDecomp(moving_avg)
            self.decomp2 = SeriesDecomp(moving_avg)

        self.dropout = nn.Dropout(p=1.0 - dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def construct(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)

        y = ops.transpose(x, (0, 2, 1))
        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = ops.transpose(y, (0, 2, 1))

        res, _ = self.decomp2(x + y)
        return res, attn

class Encoder(nn.Cell):
    """
    Autoformer Encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.CellList(attn_layers)
        self.conv_layers = nn.CellList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def construct(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer(nn.Cell):
    """
    Autoformer decoder layer with progressive decomposition
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1, has_bias=True)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, has_bias=True)

        if isinstance(moving_avg, list):
            self.decomp1 = SeriesDecompMulti(moving_avg)
            self.decomp2 = SeriesDecompMulti(moving_avg)
            self.decomp3 = SeriesDecompMulti(moving_avg)
        else:
            self.decomp1 = SeriesDecomp(moving_avg)
            self.decomp2 = SeriesDecomp(moving_avg)
            self.decomp3 = SeriesDecomp(moving_avg)

        self.dropout = nn.Dropout(p=1.0 - dropout)
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def construct(self, x, cross, x_mask=None, cross_mask=None):
        
        # print(f"Autoformer DecoderLayer before self_attention, x shape is {x.shape}")
        
        x = x + self.dropout(self.self_attention(x, x, x, x_mask)[0])
        x, trend1 = self.decomp1(x)
        
        # print(f"Autoformer DecoderLayer before cross_attention, x shape is {x.shape}, cross shape is {cross.shape}")

        x = x + self.dropout(self.cross_attention(x, cross, cross, cross_mask)[0])
        x, trend2 = self.decomp2(x)

        y = ops.transpose(x, (0, 2, 1))
        y = self.conv1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = ops.transpose(y, (0, 2, 1))

        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = ops.transpose(self.projection(ops.transpose(residual_trend, (0, 2, 1))), (0, 2, 1))

        return x, residual_trend

class Decoder(nn.Cell):
    """
    Autoformer Decoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.CellList(layers)
        self.norm = norm_layer
        self.projection = projection

    def construct(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask, cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend
