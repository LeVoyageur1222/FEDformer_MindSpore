import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class Transformer(nn.Cell):
    """
    Vanilla Transformer (MindSpore version)
    """

    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=configs.enc_in, d_model=configs.d_model, embed_type=configs.embed,
            freq=configs.freq, dropout=configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            c_in=configs.dec_in, d_model=configs.d_model, embed_type=configs.embed,
            freq=configs.freq, dropout=configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, factor=configs.factor,
                                      attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm((configs.d_model,))
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    self_attention=AttentionLayer(
                        FullAttention(mask_flag=True, factor=configs.factor,
                                      attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    cross_attention=AttentionLayer(
                        FullAttention(mask_flag=False, factor=configs.factor,
                                      attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm((configs.d_model,)),
            projection=nn.Dense(configs.d_model, configs.c_out)
        )

    def construct(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                  enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Encoder input
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder input
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
