
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time

class CausalSelfAttention(nn.Module):
    """In parallel multiple heads/streams, outputs concatenated"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value batched for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # a flag for scaling initialization to compensate for increase in variance due to residual connections (variance grows if I keep summing)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, following OpenAI naming
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size))
            .view(1,1,config.block_size,config.block_size)
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

        # FlashAttention: fuse kernels for attention
        # Does more flops (tradeoff) but because of operator fusion, much faster
        # In particular, att never gets written to global memory (T x T = 1024 * 1024)

        # # attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # only attend to historic tokens
        # att = F.softmax(att, dim=-1) # normalize attention
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Just have Pytorch use FlashAttention for us.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


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
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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
    block_size: int = 1024
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 eot token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimensionality

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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # copies the data pointer

        # param initialization
        self.apply(self._init_weights) # apply iterates all submodules
    
    def _init_weights(self, module):
        # No need to change default initialization of LayerNorm
        if isinstance(module, nn.Linear):
            stdConfig = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 2x because we have two linears per layer: block.attn and block.mlp
                stdConfig *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=stdConfig)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is token indices
        # idx is of shape (B, T)
        B,T = idx.size()
        assert T <= self.config.block_size, f"cannott forward sequence of length {T}, block size is {self.config.block_size}"

        # forward token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings shape (T, n_embd) # same for every row, then broadcast
        tok_emb = self.transformer.wte(idx) # token embeddings shape (B, T, n_embd)
        x = tok_emb + pos_emb # combine token and position embeddings

        # now forward through transformer
        for block in self.transformer.h:
            x = block(x)
        # forward final layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # shape of input to x-entropy is B*T x V, B*T x 1
            # recall: logits are basically log counts
            # softmax = logits.exp (counts) / logits.exp (counts).sum(dim=-1, keepdim=True), i.e. normalized counts
            # cross entropy loss is just -log(pr[target])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head, and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
        }[model_type]

        config_args['vocab_size'] = 50257 # from GPT
        config_args['block_size'] = 1024 # from GPT

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init hugging face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for Conv1D weights that are transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        print("successful loaded gpt2 pretrained weights")
        return model
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        return flops_achieved

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # we only weight-decay the weights that participate in matrix multiplications
        # want to do weight decay for regularization...
        # i.e. only weight decay all weight tensors in matmuls + embeddings decay; biases and layer norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # create AdamW optimizer and use the fused version if it is available
        # Fused just fuse all the kernels, so a single time, for all the parameters, call a kernel that updates them (morally)
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) 

        return optimizer


class DataLoaderLite:
    """Note, we skip the ramp up in batch size from GPT-3, it might only be slightly useful. (Why might ramp up be useful? Because early first order stuff (biases, which tokens should be skipped, etc) don't need large batches to learn (todo aern't datapoints in a batch independent anyways though).)"""

    reuseDict = {}
    lastBatchPosition = 0

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # notate that we've used this batch
        self.reuseDict[self.current_position] = self.reuseDict.get(self.current_position, 0) + 1
        self.lastBatchPosition = self.current_position

        # advance current position in the tensor
        self.current_position += B*T
        # if loading next batch is out of bounds, reset
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y
    
    def num_times_latest_batch_used(self):
        return self.reuseDict[self.lastBatchPosition]

num_return_sequences = 2
max_length = 30

# Detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
# device = "cpu" # override
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# get a data batch
# B = Batch size, T = Time
# if it doesn't fit, GPU out of memory, reduce batch size by powers of 2
train_loader = DataLoaderLite(B=16, T=1024)

# reduce precision requirement to use TensorFloat32 datatype
torch.set_float32_matmul_precision("high")

# get logits
# Override vocab size --- to be closer to a nice power of 2, because of how kerenels/GPT work.
model = GPT(GPTConfig(vocab_size=50304)) # model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)
print("compilation is on")
model = torch.compile(model) # use pytorch heavy lifting
# logits, loss = model(x, y)
# print(loss) # (B, T, vocab_size)

# Implement cosine lr decay in the style of GPT-3
max_lr = 6e-4 # from GPT 3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 500

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        # make sure not to start at 0, else useless iteration
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1 # sanity check
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1, goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# Learn
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # stealing hyperparams from gpt3

for step in range(max_steps):
    t0 = time.time()
    # each datapoint used only once
    x, y = train_loader.next_batch()
    timesBatchUsed = train_loader.num_times_latest_batch_used()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad() # recall that .backwards() adds to gradients in pytorch, so must start at 0

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()

    # clip the global norm of the gradient at 1.0 (i.e. grad = grad / grad.norm)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # ^ rationale: really high loss (i.e. unlucky batch) may cause a really high gradient and shock the model out of optimal path; a little bit hacky.
    # ^ returns norm of the gradient.

    # Learning rate changes dynamically; (cosine lr decay)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        # apparently this is how you set the lr in pytorch
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize() # for timing, wait for workload to finish
    t1 = time.time()
    dt = (t1 - t0)*1000 # time difference in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    # flops = model.estimate_mfu(fwdbwd_per_iter=(train_loader.B * train_loader.T), dt=dt)
    flops = 0
    print(f"step {step}, loss: {loss.item():.8f}, lr:{lr:.4e}, norm:{norm:.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}, flops:{flops / 1e12:.2f}, batch-reuse:{timesBatchUsed}") # .item() ships value from device back to host 


enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

torch.manual_seed(42)
# torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits, _ = model(x) # (B, T, vocab_size)
        # take logits at the last location
        logits = logits[:,-1,:] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabiliities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the correspdongin indicies
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

