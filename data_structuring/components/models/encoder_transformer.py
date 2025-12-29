from torch import nn


class EncoderTransformer(nn.TransformerEncoder):
    """A simple encoder transformer that stacks `depth` transformer encoder layers.
    Refer to [torch.nn.TransformerEncoder]
    (https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer)
    """

    def __init__(self,
                 d_model: int = 64,
                 nhead: int = 2,
                 mlp_ratio: float = 2.0,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 bias: bool = True,
                 depth: int = 1,
                 *args,
                 **kwargs):
        # Create one encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=int(d_model * mlp_ratio),
                                                   dropout=dropout,
                                                   activation=activation,
                                                   layer_norm_eps=layer_norm_eps,
                                                   batch_first=batch_first,
                                                   norm_first=norm_first,
                                                   bias=bias)

        # Merge `depth` layers to form the encoder
        super().__init__(encoder_layer, depth, *args, **kwargs)
