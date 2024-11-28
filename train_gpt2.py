
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """In parallel multiple heads/streams, outputs concatenated"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value batched for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, following OpenAI naming
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(1,1,config)
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", C is number of channels is nh * hs
        # e.g. GPT2 (124M), n_head = 12, hs = 64, nh*hs=C=768 channels
        # each token emits three vectors query, key, value
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        # treat heads as batches
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # only attend to historic tokens
        att = F.softmax(att, dim=-1) # normalize attention
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # note residual connection x
        x = x + self.attn(self.ln_1(x))  # reduce operation (all to all)
        x = x + self.mlp(self.ln_2(x))  # map operation (one to one)
        return x


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # weights of token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # weights of position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # weights of layer normalization
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
