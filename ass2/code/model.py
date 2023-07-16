import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import *


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, config, mat):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.embed_dim = config["embed_dim"]
        self.use_glove = config["use_glove"]
        if self.use_glove:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(mat).float(), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, config, mat, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_dim = config["hidden_dim"]
        self.embed_dim = config["embed_dim"]
        self.max_len = config['max_len']
        self.use_glove = config["use_glove"]
        self.attn_type = config["attn_type"]

        if self.use_glove:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(mat).float(), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)

        self.attn = nn.Linear(self.hidden_dim * 2, self.max_len)
        self.W = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.W1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.V = nn.Linear(self.hidden_dim, 1)

        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        # 1 x emb -> 1 x 1 x emb
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        if self.attn_type == 0:
            # 1. cat input yt and prev_state st-1 to [yt st-1]
            # 2. linear transform [yt st-1] to row vector (1 x max_len) and softmax it
            attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        elif self.attn_type == 1:  # multiplicative attention
            attn_weights = torch.mm(self.W(encoder_outputs), torch.cat((embedded[0], hidden[0]), 1).t())
            attn_weights = F.softmax(attn_weights.t(), dim=1)
        else:  # additive attention
            attn_weights = torch.cat((embedded[0], hidden[0]), 1).repeat(self.max_len, 1)
            attn_weights = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(attn_weights)))
            attn_weights = F.softmax(attn_weights.t(), dim=1)

        # take attention weighted sum of the encoder outputs
        # get a 1 x 1 x hid_dim tensor, i.e. the context vector of an input sentence
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # cat input yt and context vector c to [yt c]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # linear transform and relu [yt c] to get a new input y't
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        # feed new input y't and prev_state st-1 to gru
        output, hidden = self.gru(output, hidden)
        # linear transform current output dt to d't (hid_dim to vocab_size)
        # log softmax it to get distribution
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

