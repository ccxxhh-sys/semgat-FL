import torch.nn as nn
from gatv2 import GATv2Layer
from gcnn import GCNN
from SubLayerConnection import SublayerConnection
from LayerNorm import LayerNorm

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, use_gcnn=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.use_gcnn = use_gcnn
        if self.use_gcnn:
            self.Tconv_forward = GCNN(dmodel=hidden)
        else:
            self.Tconv_forward = GATv2Layer(dmodel=hidden, n_heads=attn_heads, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask, inputP, node_type):
        #x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        #x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, pos))
        #x = self.sublayer3(x, lambda _x: self.combination2.forward(_x, _x, charem))
        #print(x.size())
        gate_holder = {}
        def _body(_x):
            if self.use_gcnn:
                out = self.Tconv_forward.forward(_x, None, inputP)
                gate_holder["l1"] = _x.new_zeros(_x.size(0))
                return out
            out, gate_l1 = self.Tconv_forward.forward(_x, inputP, node_type)
            gate_holder["l1"] = gate_l1
            return out
        x = self.sublayer4(x, _body)
        x = self.norm(x)
        gate_l1 = gate_holder.get("l1", None)
        if gate_l1 is None:
            gate_l1 = x.new_zeros(x.size(0))
        return self.dropout(x), gate_l1
