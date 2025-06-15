import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from layers.Autoformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, MyLayerNorm, SeriesDecomp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Embed import DataEmbedding_wo_pos


class Autoformer(nn.Cell):
    """
    Autoformer (MindSpore version)
    - Introduces series decomposition + autocorrelation attention
    """

    def __init__(self, configs):
        super(Autoformer, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomposition
        kernel_size = configs.moving_avg
        self.decomp = SeriesDecomp(kernel_size[0]) if isinstance(kernel_size, list) else SeriesDecomp(kernel_size)

        # Embedding (no positional encoding)
        self.enc_embedding = DataEmbedding_wo_pos(
            c_in=configs.enc_in, d_model=configs.d_model,
            embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            c_in=configs.dec_in, d_model=configs.d_model,
            embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False, factor=configs.factor,
                                        attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=MyLayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=True, factor=configs.factor,
                                        attention_dropout=configs.dropout, output_attention=False),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    cross_attention=AutoCorrelationLayer(
                        AutoCorrelation(mask_flag=False, factor=configs.factor,
                                        attention_dropout=configs.dropout, output_attention=False),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=MyLayerNorm(configs.d_model),
            projection=nn.Dense(configs.d_model, configs.c_out)
        )

    def construct(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                  enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Seasonal-Trend decomposition for encoder input
        seasonal_init, trend_init = self.decomp(x_enc)

        # Prepare decoder seasonal/trend input
        mean = x_enc.mean(axis=1, keep_dims=True).repeat(self.pred_len, axis=1)
        zeros = ops.zeros((x_dec.shape[0], self.pred_len, x_dec.shape[2]))

        trend_init = ops.concat([trend_init[:, -self.label_len:, :], mean], axis=1)
        seasonal_init = ops.concat([seasonal_init[:, -self.label_len:, :], zeros], axis=1)

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )

        # Final output
        dec_out = seasonal_part + trend_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
