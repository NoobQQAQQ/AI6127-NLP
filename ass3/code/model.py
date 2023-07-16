import math
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, no_scale=False, mask=None, dropout=None):
    # query shape: B x h x L x d_k
    # mask shape: Lx1XL
    d_k = query.size(-1)
    if no_scale:
        scores = torch.matmul(query, key.transpose(-2, -1))
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
    # scores shape: B x h x L x L
    if mask is not None:
        scores = scores + mask
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return shape: B x h x L x d_k
    return torch.matmul(p_attn, value), p_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, no_scale=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.no_scale = no_scale

    def forward(self, query, key, value, mask=None):
        # LxBxE -> BxLxE
        query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # BxLxE -> B x h x L x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, self.no_scale, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # BxLxE -> LxBxE
        x = x.transpose(0, 1).contiguous()
        return self.linears[-1](x)


class TransformerLMEncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(TransformerLMEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=cfg.emb_dim, h=cfg.nhead,
                                            no_scale=cfg.no_scale, dropout=cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.emb_dim)
        self.dropout1 = nn.Dropout(p=cfg.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.emb_dim, cfg.ffn_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.emb_dim)
        )
        self.norm2 = nn.LayerNorm(cfg.emb_dim)
        self.dropout2 = nn.Dropout(p=cfg.dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.attention(x, x, x, mask))
        x = self.dropout1(x)

        x = self.norm2(x + self.ffn(x))
        x = self.dropout2(x)
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg, ntoken):
        super(TransformerLM, self).__init__()
        self.d_model = cfg.emb_dim
        self.embedding = nn.Embedding(ntoken, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, cfg.dropout)

        encoder_layer = TransformerLMEncoderLayer(cfg)
        self.transformer_encoder = clones(encoder_layer, cfg.N)

        self.decoder = nn.Linear(self.d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        sz = len(src)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)
        for layer in self.transformer_encoder:
            src = layer(src, src_mask)
        output = self.decoder(src)
        return F.log_softmax(output, dim=-1)
