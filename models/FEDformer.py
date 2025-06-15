import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from layers.Autoformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer, MyLayerNorm, SeriesDecomp, SeriesDecompMulti
from layers.AutoCorrelation import AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletTransform, MultiWaveletCross
from layers.Embed import DataEmbedding_wo_pos

class FEDformer(nn.Cell):
    """
    FEDformer (MindSpore version)
    - Frequency Enhanced Decomposed Transformer
    """

    def __init__(self, configs):
        super(FEDformer, self).__init__()
        self.version = configs.version  # 'Fourier' or 'Wavelets'
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomposition
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = SeriesDecompMulti(kernel_size)
        else:
            self.decomp = SeriesDecomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            c_in=configs.enc_in, d_model=configs.d_model,
            embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            c_in=configs.dec_in, d_model=configs.d_model,
            embed_type=configs.embed, freq=configs.freq, dropout=configs.dropout
        )

        # Encoder/Decoder Attention choice
        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len_q=self.seq_len//2+self.pred_len, seq_len_kv=self.seq_len,
                modes=configs.modes, ich=configs.d_model,
                base=configs.base, activation=configs.cross_activation
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len=self.seq_len, modes=configs.modes, mode_select_method=configs.mode_select
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len=self.seq_len//2+self.pred_len, modes=configs.modes, mode_select_method=configs.mode_select
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len_q=self.seq_len//2+self.pred_len, seq_len_kv=self.seq_len,
                modes=configs.modes, mode_select_method=configs.mode_select
            )

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
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
                        decoder_self_att,
                        d_model=configs.d_model,
                        n_heads=configs.n_heads
                    ),
                    cross_attention=AutoCorrelationLayer(
                        decoder_cross_att,
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

        # Seasonal-Trend decomposition
        seasonal_init, trend_init = self.decomp(x_enc)

        mean = x_enc.mean(axis=1, keep_dims=True).repeat(self.pred_len, axis=1)
        zeros = ops.zeros((x_dec.shape[0], self.pred_len, x_dec.shape[2]))

        trend_init = ops.concat([trend_init[:, -self.label_len:, :], mean], axis=1)
        seasonal_init = ops.concat([seasonal_init[:, -self.label_len:, :], zeros], axis=1)

        # Embedding
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
