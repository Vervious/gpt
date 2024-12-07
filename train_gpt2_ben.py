
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time
import hellaswag

# The general idea:
# input stream is not tokens, but embedded tokens
# evaluate a single layer
# compute loss for the previous layer using confidence of current layer
# have some confidence threshold for "what we consider external output"
# if previous layer was already very confident (i.e. "output something"), then evaluate that output against the true target

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
        # Don't need this once we switched to flashattention
        #  # not really a 'bias', more of a mask, following OpenAI naming
        # self.register_buffer("bias", torch.tril(
        #     torch.ones(config.block_size, config.block_size))
        #     .view(1,1,config.block_size,config.block_size)
        # )

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

        # Note: C = n_embd
        # head size = C // self.n_head = 64

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
        # (B, T, n_embd)
        # output projection
        y = self.c_proj(y) # NOTE: what is the point of this
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
        out = self.ln_1(x)
        x = x + self.attn(self.ln_1(x))  # reduce operation (all to all)
        x = x + self.mlp(self.ln_2(x))  # map operation (one to one)
        return x, out


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 eot token
    n_layer: int = 12 # number of layers # NOTE: layers become redundant in my architecture
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
            # TODO: have the KQV weights also be shared across layers
            # h = nn.ModuleList([shared_block for _ in range(config.n_layer)]),
            sharedblock = Block(config), # NOTE: this does not seem to degrade performance at least early in the training process
            # weights of layer normalization
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

    def forward(self, idx, targets=None, all_logits=False):
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
        loss = 0.0
        allLogits = []
        # print(self.transformer.h)
        for i in range(self.config.n_layer):
            x, _out_embedded_tokens_from_prev = self.transformer.sharedblock(x)
            # NOTE: _out_embedded_tokens_from_prev is post layer norm, so don't have to compute layer norm again. In principle this computes loss for previous weight application, but in reality it all gets mishmashed together.
            # _out_embedded_tokens_from_prev should be (B, T, n_embd)
            _logits = self.lm_head(_out_embedded_tokens_from_prev) # (B, T, vocab_size)
            _targets = _logits.detach().max(dim=-1)[1]
            # _logprobs = F.log_softmax(_logits, dim=-1) # self.softmax(_logits).clamp(min=1e-9, max=1-1e-9) # (B, T, vocab_size)
            # (_logprobs.exp() * _logprobs).sum(dim=-1)
            # _block_loss = -1 * _logprobs.min(dim=-1)[0].mean()
            _block_loss = F.cross_entropy(_logits.view(-1, _logits.size(-1)), _targets.view(-1))
            loss += _block_loss # NOTE: just try this for now
            if all_logits:
                allLogits.append(_logits)
        # print(len(self.transformer.h))

        # forward first layer norm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if all_logits:
            allLogits.append(logits)
        trueLoss = None
        if targets is not None:
            # shape of input to x-entropy is B*T x V, B*T x 1
            # recall: logits are basically log counts
            # softmax = logits.exp (counts) / logits.exp (counts).sum(dim=-1, keepdim=True), i.e. normalized counts
            # cross entropy loss is just -log(pr[target])
            # F.cross_entropy takes average over B*T
            xe = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # print("LOSS", xe.item())
            loss += xe
            trueLoss = xe.detach()
        if all_logits:
            return allLogits, loss, trueLoss
        else:
            return logits, loss, trueLoss

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
        N = sum(p.numel() for p in self.parameters())
        N -= self.transformer.wpe.weight.numel() # due to parameter sharing, I think we should get rid of this?

        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/(dt*1000))
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
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            with open(log_file, "a") as f:
                f.write(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                f.write(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    

        # create AdamW optimizer and use the fused version if it is available
        # Fused just fuse all the kernels, so a single time, for all the parameters, call a kernel that updates them (morally)
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) 

        return optimizer

import numpy as np



class DataLoaderLite:
    """Note, we skip the ramp up in batch size from GPT-3, it might only be slightly useful. (Why might ramp up be useful? Because early first order stuff (biases, which tokens should be skipped, etc) don't need large batches to learn (todo aern't datapoints in a batch independent anyways though).)"""

    reuseDict = {}
    lastBatchPosition = 0
    SHAKESPEARE = False

    def load_tokens(self, filename):
        if self.SHAKESPEARE:
            with open(filename, 'r') as f:
                text = f.read()
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(text)
            ptt = torch.tensor(tokens)
        else:
            # has already been tokenized for us using gpt2 tokenizer
            npt = np.load(filename)
            npt = npt.astype(np.int32) # see Kaparthy 12c1ea8
            ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}


        if self.SHAKESPEARE:
            # shakespeare
            self.shards = ['shakespeare.txt']
        else:
            # FineWeb
            # get the shard filenames
            data_root = "edu_fineweb10B"
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            if master_process:
                print(f"found {len(shards)} shards for split {split}")     
        
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        # allocate gpus to different starting locations of the data
        self.current_position = self.B * self.T * self.process_rank

        if master_process:
            print(f"{self.split}: loaded {len(self.tokens)} tokens (first shard)")
            print(f"{self.split}: 1 epoch (1 shard) = {len(self.tokens) // (self.B*self.T)} mini-batches")


    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets

        # notate that we've used this batch
        self.lastBatchPosition = (self.current_shard, self.current_position)
        self.reuseDict[self.lastBatchPosition] = self.reuseDict.get(self.lastBatchPosition, 0) + 1

        # advance current position in the tensor
        self.current_position += B*T * self.num_processes
        # if loading next batch is out of bounds, advance to beginning of next shard
        # (if only one shard, i.e. in shakespeare, go to the beginning of that same shard)
        if self.current_position + B*T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    
    def num_times_latest_batch_used(self):
        return self.reuseDict[self.lastBatchPosition]



# Distributed GPUs
# ===================
# Simple launch:
# python train_gpt2.py
# DDP Launch:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run (sketchy)
# ddp = False # override
if ddp:
    assert torch.cuda.is_available(), "torch.distributed only works with CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # 0 to 7
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # used in multi node setting, rank of GPU on single node, 0 to 7. Unused by us?
    ddp_world_size = int(os.environ['WORLD_SIZE']) # 8 for 8 GPUs
    device = f'cuda:{ddp_local_rank}' # each process only sees one GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this is the leader doing logging, checkpoint, etc
else:
    # vanilla, non-DPP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
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

# ===================
# HYPERPARAMETERS
# ===================


# Create log and persistence directory
log_dir = "log-ben"
sample_dir = "samples-ben"
checkpoint_dir = "checkpoints-ben"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
sample_file = os.path.join(sample_dir, "main.txt")
if master_process:
    with open(log_file, "w") as f: # clear log file
        pass
    with open(sample_file, "w") as f: # clear samples file
        pass

# We want a larger batch size to follow GPT-3 Small, roughly B*T = 0.5M; but setting B = 488 will blow up the GPU.
# Since we only have small GPUs, we'll just simulate large batches using accumulation.
B = 8 # micro batch size, will do forward backward but not do an update yet # previously 16 # A100 can do 64?
T = 1024 # sequence length # 1024
total_batch_size = 2 * 16 * 1024# 524288 # B*T # TODO change to 524288 # 2**19 ~0.5M in number of tokens
max_steps = 300000 + 1 # How many steps do we train for
# Implement cosine lr decay in the style of GPT-3
max_lr = 2*6e-4 # from GPT 3 paper # double it because it seems to work
min_lr = max_lr * 0.1
warmup_steps = 10
use_compile = False # May run into bugs

hello_swag_frequency = 20000
validation_frequency = 2000
checkpoint_frequency = 20000
sample_frequency = 100

assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*(# gpus)"
grad_accum_steps = total_batch_size // (B*T * ddp_world_size) # 4; so each batch split into 4 mini batches
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"Mini-batch size: {B}*{T}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"Training max steps: {max_steps}")


    with open(log_file, "a") as f:
        f.write(f"total desired batch size: {total_batch_size}")
        f.write(f"Mini-batch size: {B}*{T}")
        f.write(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        f.write(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print("I am GPU", ddp_rank, " of ", ddp_world_size)

# get a data batch
# B = Batch size, T = Time
# if it doesn't fit, GPU out of memory, reduce batch size by powers of 2
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# reduce precision requirement to use TensorFloat32 datatype
torch.set_float32_matmul_precision("high")

# get logits
# Override vocab size --- to be closer to a nice power of 2, because of how kerenels/GPT work.
model = GPT(GPTConfig(vocab_size=50304, block_size=T)) # model = GPT.from_pretrained('gpt2')
# model.eval()
model.to(device)
if use_compile:
    print("compilation is on")
    model = torch.compile(model) # use pytorch heavy lifting
else:
    print("compilation is off")
if ddp:
    # docs unclear according to kaparthy, but he thinks it should be ddp_local_rank
    # Wrapper:
    # - nothing changes in forward pass
    # - in backwards pass, gradients are averaged across all GPUs (via allreduce)
    # -- every single rank gets the output of the allreduce
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model



# # ================ FineWeb
# warmup_steps = 100 # 375e6 tokens (GPT3) is about 715 steps counting 2**19 tokens per step, but set to 100 since it doesn't really matter
# max_steps = 19073 # use each token only once approximately; we do 2**19 tokens per step (total batch size) --- we want to do roughly 10B (that is number of unique tokens in dataset)... 10B / 2**19 ~ 19073

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
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) # stealing hyperparams from gpt3

enc = tiktoken.get_encoding('gpt2') # for following along progress of training


for step in range(max_steps):
    t0 = time.time()
    
    # get validation loss once in a while (I think this is a bit sketchy)
    if step > 0 and step % validation_frequency == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)  
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _, loss = model(x, y)
                loss = loss / val_loss_steps # average accumulated loss
                val_loss_accum += loss.detach() # why do i need to call detach if I never call backward on it
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
    
    if ((step % checkpoint_frequency == 0 and step > 0) or step == max_steps - 1):
        # save model checkpoint
        if master_process:
            print(f"saving model checkpoint: {checkpoint_dir}/model_{step}.pt")
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_{step}.pt")

    if ((step % hello_swag_frequency == 0 and step > 0) or step == max_steps - 1) and (not use_compile):
        # eval on hello swag
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(hellaswag.iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            # render example into tokens and labels
            _, tokens, mask, label = hellaswag.render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _, loss = model(tokens)
                pred_norm = hellaswag.get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # Reduce stats
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hellaswag {acc_norm:.4f}\n")
    
    # Also generate some sample text once in a while
    if (step > max_steps - 2 and (not use_compile)) or (step % sample_frequency == 50 and (not use_compile)) or step == 1: # > 0 and step % 100 == 0: # Like Kaparthy, I run into compilation issues
        model.eval()
        num_return_sequences = 1
        max_length = 128
        tokens = enc.encode("Hello, I'm a language model,")
        printgen = tokens
        leftParens = enc.encode("\n\t\t(")
        rightParens = enc.encode(")\n")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # different seed for each GPU
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _, _ = model(xgen, all_logits=True) # (B, T, vocab_size)
                    
                for _logit in logits:
                    # take the last token logits
                    printgen.extend(leftParens)
                    for j in range(5):
                        _logit = _logit[:,-(j+1),:]
                        _probs = F.softmax(_logit, dim=-1)
                        vals, idxs = _probs.max(dim=-1, keepdim=True) #(B, 1)
                        printgen.append(idxs[0,0].item())
                        printgen.extend(enc.encode(f":{vals[0,0].item():.4f}"))
                    printgen.extend(rightParens)

                logits = logits[-1]
                # take logits at the last location
                logits = logits[:,-1,:] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabiliities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the correspdongin indicies
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
                printgen.append(xcol[0,0].item())
        # print the generated text
        for i in range(num_return_sequences):
            tokens = printgen
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
            with open(sample_file, "a") as f:
                f.write(f"{step}: sample {i}: {decoded}\n")
        
    # Actual training loop
    model.train()
    optimizer.zero_grad() # recall that .backwards() adds to gradients in pytorch, so must start at 0
    loss_accum = 0.0
    loss_accum_all = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        timesBatchUsed = train_loader.num_times_latest_batch_used()
        x, y = x.to(device), y.to(device)
        if ddp:
            # (Kaparthy says: hope this isn't a breaking torch change, should maybe use no_sync)
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss, trueloss = model(x, y)
        # Need to scale loss by grad_accum_steps because we need to get the average loss over the entire batch (not just over mini batches)
        loss = loss / grad_accum_steps
        loss_accum += trueloss.detach() / grad_accum_steps
        loss_accum_all += loss.detach()
        # .backward() will just += the gradient
        # DDP will automatically allreduce this for us
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

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
    if master_process:
        dt = (t1 - t0)
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        flops = raw_model.estimate_mfu(fwdbwd_per_iter=(tokens_processed), dt=dt)

        print(f"step {step}, loss: {loss_accum.item():.8f}, allloss: {loss_accum_all.item():.8f} lr:{lr:.4e}, norm:{norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}, flops:{flops / 1e12:.2f}, batch-reuse:{timesBatchUsed}") # .item() ships value from device back to host 
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.8f}, allloss: {loss_accum_all.item():.4f}, lr:{lr:.4e}, norm:{norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}, flops:{flops / 1e12:.2f}, batch-reuse:{timesBatchUsed}\n")


if ddp:
    destroy_process_group()
exit(0)


# if not master_process:
#     if ddp:
#         destroy_process_group()
#     exit(0)


# print("Generating text...")
# num_return_sequences = 2
# max_length = 100

# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# torch.manual_seed(42)
# # torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get logits
#     with torch.no_grad():
#         logits, _ = model(x) # (B, T, vocab_size)
#         # take logits at the last location
#         logits = logits[:,-1,:] # (B, vocab_size)
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         # select a token from the top-k probabiliities
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the correspdongin indicies
#         xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#         # append to the sequence
#         x = torch.cat((x, xcol), dim=1)

# # print generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)


# if ddp:
#     destroy_process_group()
