
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time
import hellaswag

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )


class DualModule(nn.Module):

    def __init__(self):
        super(DualModule, self).__init__()



# The general idea:
# input stream is not tokens, but embedded tokens
# evaluate a single layer
# compute loss for the previous layer using confidence of current layer
# have some confidence threshold for "what we consider external output"
# if previous layer was already very confident (i.e. "output something"), then evaluate that output against the true target

class CausalSelfAttention(DualModule):
    """In parallel multiple heads/streams, outputs concatenated"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value batched for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
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
        # q,k = qkv.split(self.n_embd, dim=2)
        # treat heads as batches
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # # NOTE: value matrix seems redundant, could just change to the below
        # why do we split up dimensionality of x this way
        # v = torch.eye(T, device=x.device).view(1, 1, T, T) # (B, nh, T, T)
        # v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # TODO figure out what is wrong with the below code and why it breaks training if we use it instead of the pytorch version
        # # Note: C = n_embd
        # # head size = C // self.n_head = 64

        # # FlashAttention: fuse kernels for attention
        # # Does more flops (tradeoff) but because of operator fusion, much faster
        # # In particular, att never gets written to global memory (T x T = 1024 * 1024)

        # # attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # Upper triangular matrix with diagonal offset by 1
        # mask = mask.unsqueeze(0).unsqueeze(0)

        # # mask = self.bias[:, :, :T, :T] == 0
        # att = att.masked_fill(mask, float('-inf')) # only attend to historic tokens
        # att = F.softmax(att, dim=-1) # normalize attention
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Just have Pytorch use FlashAttention for us. #TODO NVM
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 


        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side
        # (B, T, n_embd)
        # output projection
        y = self.c_proj(y) # NOTE: what is the point of this (to support dimension reduction from before, i don't think we actualy need to do dimension reduction)
        return y


class MLP(DualModule):

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


class Block(DualModule):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
        self.attn = CausalSelfAttention(config)
        self.attn2 = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
        self.mlp = MLP(config)
        self.mlp2 = MLP(config)
    
    def forward(self, x, res, y):
        # note residual connection res
        # out = self.ln_1(x)
        # NOTE, for some reason LN(x) + self.attn(LN(x)) doesn't work as well. 
        # ^ it is incredibly important that the residual is not passed through the layer norm... (TODO why??? Layers can no-op?)
        # x = x + self.attn(self.ln_1(x))  # reduce operation (all to all)
        # NOTE: res will generally be very large...
        y = self.attn(x)*self.mlp(x) # res + self.attn(x) # NOTE that the residual connection x is already layer normed, unlike usual transformer implementation # TODO add back residual res + . NOTE x + self.attn(x) is simply horrible (why?)... we cannot layer norm it (prev too big or too small?)
        # Maybe the layernorm just destroys relative magnitude of things...
        # NOTE, likewise LN(x) + mlp(LN(x)) doesn't work as well? The residual literally has to be untouched. 
        midx = y
        x = res + y  # + self.mlp(x)  # map operation (one to one) # TODO ADD OR MULTIPLY? x + self.mlp #TODO additive attn, multiplicative mlp
        newres = x
        x = self.ln_1(x) # TODO UNCOMMENT, and then try removing res (applying attn to res)
        # NOTE: NEXT is usually ~1.0, prev is usually < 1.0 early on.
        # NOTE: By step 116, prev is much larger, ~2.0
        # also seems to get bigger as layer number grows...
        # so the layer norm really makes things bigger.
        # NOTE: if x + self.mlp, prev grows much much quicker, ~30 by step 48
        # ^ but it seems to get a bit smaller over time
        # Large (and increasing over layer num) prev remains true even if we don't 
        # compute loss for every layer. (This is probably due to reuse...)
        # NOTE: growth continues even if transformer block is not reused
        # ^ again, in vanilla GPT, it gets smaller over time.
        # NOTE: for some reason, for res*attn(x) with x*mlp(LN(x)), std is always 0 even for alllayer
        # NOTE: doing res * attn(x) and x + mlp(LN(x)) explodes the prev when training last layer only, but when doing all layer loss, it is reasonable... (why?)

        # NOTE seems quite good res + attn(x), comment out ln_1
        return x, newres, midx, y


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 eot token
    n_layer: int = 12 # number of layers # NOTE: layers become redundant in my architecture
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimensionality

class GPT(DualModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        sharedBlock = Block(config)

        self.transformer = nn.ModuleDict(dict(
            # weights of token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # weights of position embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #  COMMEZNT
            sharedblock = sharedBlock, # NOTE: this does not seem to degrade performance at least early in the training process
            # weights of layer normalization
            ln_f = sharedBlock.ln_1, # nn.LayerNorm(config.n_embd),
            # NOTE we share ALL layer norms which may not be necessarily wise
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

    def forward(self, idx, targets=None, all_logits=False, print_weights=False):
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
        losses = torch.tensor(0.0, device=idx.device)
        confLoss = torch.tensor(0.0, device=idx.device)
        targetLoss = torch.tensor(0.0, device=idx.device)
        allLogits = []
        # print(self.transformer.h)
        res = x
        x = self.transformer.ln_f(x) # apply the initial layer norm (this is backwards from the usual implementation, but same semantically)

        _mask_BT = torch.ones(B,T,1, device=idx.device) # (B, T, 1) 
        trueLoss = None

        _true_logits = torch.zeros(B, T, self.config.vocab_size, device=idx.device)

        savedConf = []
        # savedXeFactor = []

        # _xe_prev = None
        # _just_triggered_prev = None
        _mask_BT_prev = None

        early_stop_num = 0.0
        earlyStopLayerDict = torch.zeros(self.config.n_layer, device=idx.device)

        y = x

        for i in range(self.config.n_layer):
            prev_x = x # (B, T, n_embd)

            

            
            # block = self.transformer.h[i]
            # x, res = block.requires_grad_(True)(x, res)
            x, res, midx, y = self.transformer.sharedblock.requires_grad_(True)(x, res, y) # TODO UNCOMMENT

            # # For now, try computing dot products in embedding space
            # target_embedding = torch.roll(prev_x, shifts=-1, dims=1) # shift in T dimension

            # # NOTE: ignore the far right-most one
            # dp = -1* (target_embedding[:,:-1,:] * x[:,:-1,:]).sum(dim=-1) / self.config.n_embd # dot product
            # dp = dp.mean()
            # losses += dp
            # confLoss += dp


            with torch.no_grad():
                if all_logits or print_weights:
                    _logits = self.lm_head(x)
                    # _logits_target_ = self.lm_head(x[:,:-1,:])
                    # _target_embd = self.lm_head(target_embedding[:,:-1,:])
                if all_logits:
                    allLogits.append(_logits)
                if print_weights:
                    prevsize = res.std(dim=-1).mean()
                    nextsize = x.std(dim=-1).mean()
                    midsize = midx.std(dim=-1).mean()
                    bprint(f"SIZE COMPARISON nextres {prevsize} attn*mlp {midsize} layernormed {nextsize}")
                # if print_weights:
                #     _batch_0_target = _target_embd[:,-7:,:] #(B, 7, vocab_size?)
                #     _probs = F.softmax(_batch_0_target, dim=-1) #(B, 7, vocab_size?)
                #     vals, idxs = _probs.max(dim=-1, keepdim=True) #(B, 7, 1)
                #     v = idxs[0,:,0].tolist()

                #     _logits_target_ = _logits_target_[:,-7:,:] #(B, 7, vocab_size?)
                #     _probs = F.softmax(_logits_target_, dim=-1) #(B, 7, vocab_size?)
                #     vals, idxs = _probs.max(dim=-1, keepdim=True) #(B, 7, 1)
                #     cur = idxs[0,:,0].tolist()

                #     a = "\t".join(enc.decode(v))
                #     b = "\t".join(enc.decode(cur))

                #     bprint(f"Layer {i} prev-targets\npre\t\t{a}\ncur\t\t{b}")

                    # if (i == 0 or i == self.config.n_layer - 2 or i == self.config.n_layer - 1 or i == self.config.n_layer) and master_process:
                    #     # print(self.transformer.sharedblock.attn.c_attn.weight.view(-1)[-6:].data)
                    #     probs = F.softmax(_logits, dim=-1) # (B, T, vocab_size?)
                    #     # print top k of the last T
                    #     # do top-k sampling of 50 (huggingface pipeline default)
                    #     # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    #     topk_probs, topk_indices = torch.topk(probs[0,-1,:], 5, dim=-1)
                    #     fstring = ["{:.5f}".format(value) for value in topk_probs.tolist()]
                    #     bprint(f"Layer {i}\n\t\t\t {fstring, enc.decode(topk_indices.tolist())}")

            if targets is not None and (i == self.config.n_layer - 1 or ALL_LAYER_LOSS):

                # NOTE: keep this because... can't return some dummy thing instead
                _logits = self.lm_head.requires_grad_(True)(x) # (B,T,Vocab_size)

                tl = F.cross_entropy(_logits.view(-1, _logits.size(-1)), targets.view(-1)) # (1)
                trueLoss = tl
                losses = losses + tl
                allLogits.append(_logits)
                

            if ENABLE_LAYER_LOSS:

                x_False, _, _, _ = self.transformer.sharedblock.requires_grad_(False)(x, res, y)
                # _logits is after application of the current layer
                _logits_incl_curr_layer = self.lm_head.requires_grad_(True)(x)
                _logits_skipping_curr_layer = self.lm_head.requires_grad_(False)(x_False) # (B, T, vocab_size)
                # TODO: seems not to work if I dont' call requires_grad(False) on lm_head (maybe for obvious reasons), now check if I do call it.
                if all_logits or i == self.config.n_layer - 1:
                    allLogits.append(_logits_skipping_curr_layer)

                
                _, _targets = _logits_skipping_curr_layer.detach().max(dim=-1) #(B, T) # NOTE: the [1] is to get the indices

                # _logprobs = F.log_softmax(_logits, dim=-1) # self.softmax(_logits).clamp(min=1e-9, max=1-1e-9) # (B, T, vocab_size)
                # (_logprobs.exp() * _logprobs).sum(dim=-1)
                # _block_loss = -1 * _logprobs.min(dim=-1)[0].mean()
                # Cross entropy returns a positive loss (i.e. - log (pr))
                _nll_max = F.cross_entropy(_logits_skipping_curr_layer.view(-1, _logits_skipping_curr_layer.size(-1)), _targets.view(-1), reduction='none') # (B*T)


                _nll_max = _nll_max.view(B,T, 1) # (B, T) = -log(pr[target])

                # print("NLLMAXSHAPE ", _nll_max.shape)

                _sample_rand = torch.rand(B, T, 1, device=idx.device).log() * -1
                # NOTE: we want to perform the usual computations perhaps, but just stop propagating the gradient? on masked values / short circuited batches / Ts.
                # I.e. for B,T where softmax(_logits).max() < _sample_rand, do an early stopping, and stop accumulating loss (where loss = -log(pr[target]) * confidence) thereafter
                # Final loss (for each B,T) is -logli under those final logits

                # mask loss where _sample_rand > _nll_max, i.e. confidence is high and a sample "hit"
                # TODO: tbd shoudl just max the loss
                mask = _sample_rand < _nll_max # True/1 if subsequent loss should be counted, False/0 otherwise # (B, T, 1)


                # if previously 0, stay 0
                # set _mask_BT: 1 if loss this layer should count, 0 otherwise (i.e. early termination)
                _new_mask_BT = mask * _mask_BT

                if i == self.config.n_layer - 1:
                    _new_mask_BT = torch.zeros_like(_mask_BT) # NOTE: always penalize last layer since we have finite number of layers

                # 1 if switched from 1 to 0 in this iteration, else 0
                _just_triggered = (_mask_BT - _new_mask_BT).detach()  #(B, T, 1)
                _mask_BT = _new_mask_BT # (B, T, 1)

                earlyStopLayerDict[i] = _just_triggered.sum().item()

                _true_logits = _true_logits + _just_triggered * _logits_incl_curr_layer # later we will backprop this against targets loss

                # _masked_logits = torch.where(mask.unsqueeze(-1), torch.zeros_like(_logits), _logits) # (B, T, vocab_size)   
                

                # _confidence = 1 - torch.exp(-1*_nll_max) # Just make it 0 to 1 linear, instead of something that grows exponentially
                _confidence = -1 * log1mexp(-_nll_max) # NOTE were we running into numerical stability problems? # (B, T, 1)
                # _confidence = -1 * torch.log1p(-1*torch.exp(-1*_nll_max))
                # _confidence = -1 * torch.log(1 - torch.exp(-1*_nll_max)) # NOTE: confidence calculation, makes loss better, see readme
                # The higher the max, the higher the confidence (to inf)
                # max = low, low confidence (to 0)
                # TODO no grad the most recent application of attention
                # Times -1 to reward low confidence.

                if _mask_BT[0,-1,0] == 0:
                    if master_process and print_weights:
                        bprint(f"\tB=0 T=-1 skipped layer {i}")

                if print_weights:
                    early_end = self.config.n_layer # can change to smaller number to "short circuit"
                    if (i == 0 or i == early_end - 2 or i == early_end - 1 or i == self.config.n_layer) and master_process:
                        # print(self.transformer.sharedblock.attn.c_attn.weight.view(-1)[-6:].data)
                        with torch.no_grad():
                            probs = F.softmax(_logits_skipping_curr_layer, dim=-1) # (B, T, vocab_size?)
                            # print top k of the last T
                            # do top-k sampling of 50 (huggingface pipeline default)
                            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                            topk_probs, topk_indices = torch.topk(probs[0,-1,:], 5, dim=-1)
                            fstring = ["{:.5f}".format(value) for value in topk_probs.tolist()]
                            bprint(f"Layer {i} B=0,T=-1 Confidence={_confidence[0,-1].item():.5f}\n\t\t\t {fstring, enc.decode(topk_indices.tolist())}")
                        # if i > 5:
                        #     break
                        # print(self.transformer.sharedblock.attn.c_attn.weight.view(-1)[-6:].data)
                    # if i == early_end - 1:
                    #     break

            
            
            
            # losses += _block_loss / self.config.n_layer # NOTE: just try this for now # TODO before or after real loss computation?
            # NOTE: naive attempt at normalizing
            # ln(0.9) = -0.1, about 90% confidence
            # so we add -0.1. if block is 50% confident, we add -0.69, reducing the loss.

            if targets is not None:
                # We want _block_loss to be high when confidence is high, and low when confidence is low... but then won't the network just output low confidence?
                # print("XE SHAPE ", xe.shape)
                # print("MASK SHAPE ", _mask_BT.shape)
                # print("CONFIDENCE SHAPE ", _confidence.shape)


                if ENABLE_LAYER_LOSS and i > 0:

                    # _target_conf = F.cross_entropy(_logits_skipping_curr_layer.view(-1, _logits_skipping_curr_layer.size(-1)), targets.view(-1), reduction='none').view(B,T, 1) # (B*T)
                    # _target_conf = -1 * log1mexp(-_target_conf)

                    # TODO make sure to scale things properly... extreme confidence gives extreme loss, maybe we should take an e^-(x)

                    # NOTE: if _just_triggered = 1, then xe is multiplied in and _confidence is ignored; else, if _just_triggered = 0, then factor = 1
                    # xe_factor = torch.pow(xe, _just_triggered)
                    # xe_factor_prev = ((_xe_prev / _confidence - 1) * _just_triggered_prev + 1)

                    # _mask_BT_prev = 0 if previous layer has triggered (inclusive)

                    # TODO UNCOMMENT THIS BLOCK
                    # NOTE: detach _mask_BT for now, I don' tthink we really need it
                    confLoss_contrib = (_confidence * _mask_BT_prev.detach()).mean() # _confidence
                    confLoss -= confLoss_contrib

                    # loss_ = (xe * _confidence * _mask_BT).mean()
                    savedConf.append(_confidence.mean())
                    # savedXeFactor.append((xe_factor_prev * _confidence).mean())
                    losses = losses - confLoss_contrib #TODO UNCMOMENT


                    # If mask is 0, then it has already "triggered", so no loss
                    # Loss is confidence unless at point of triggering
                
                # print("LOSS SHAPE", losses.shape)

                
                if ENABLE_LAYER_LOSS:
                    # _logits = self.lm_head.requires_grad_(True)(x) # (B,T,Vocab_size)
                    xe = F.cross_entropy(_logits_incl_curr_layer.view(-1, _logits_incl_curr_layer.size(-1)), targets.view(-1), reduction='none') #(B*T)
                    xe = xe.view(B, T, 1)
                    targetLoss_contrib = (xe * _just_triggered).mean()
                    losses = losses + targetLoss_contrib
                    targetLoss += targetLoss_contrib # IN PLACE
                    trueLoss = targetLoss


                if i == self.config.n_layer - 1:

                    # NOTE: keep this because... can't return some dummy thing instead

                    if not ENABLE_LAYER_LOSS:
                        pass
                        # _logits = self.lm_head.requires_grad_(True)(x) # (B,T,Vocab_size)
                        # tl = F.cross_entropy(_logits.view(-1, _logits.size(-1)), targets.view(-1)) # (1)
                        # trueLoss = tl
                        # print("lalala")
                        # losses = losses + tl
                        # allLogits.append(_logits)
                    else:
                        # # NOTE: uncomment for true trueLoss computation
                        # xe_staggered_loss = F.cross_entropy(_true_logits.view(-1, _true_logits.size(-1)), targets.view(-1))
                        # trueLoss = xe_staggered_loss 

                        # trueLoss = xe.mean().detach() #TODO compare with the prev
                        allLogits.append(_true_logits) # NOTE: later we sample from allLogits[-1]
                        # NOTE: trueLoss slightly different than targetLoss, not sure why...

                        # trueLoss = xe.mean().detach()
                        # if not ENABLE_LAYER_LOSS:
                        # _mask_BT_prev because if _mask_BT, it is always 0 at last layer

                        # BELOW COMMENTED
                        # confLoss = losses # 
                        # targetLoss = (xe * _mask_BT_prev).mean()
                        # losses = losses + targetLoss


                        # print("MASK SUM ", _mask_BT_prev.sum().item(), B*T)
                        # losses += xe.mean()
                        #NOTE for now, always penalize last layer since we have finite number of layers

                        early_stop_num = B*T - _mask_BT_prev.sum().item()


                # Used when adding loss when computing confidence of subsequent layer
                # _xe_prev = xe
                # _just_triggered_prev = _just_triggered
                _mask_BT_prev = _mask_BT

            else:
                # NOTE this code path should be unused except during inference
                if i == self.config.n_layer - 1 and all_logits:
                    if ENABLE_LAYER_LOSS:
                        allLogits.append(_true_logits)
                    else:
                        _logits = self.lm_head.requires_grad_(True)(x) # (B,T,Vocab_size)
                        allLogits.append(_logits)
                # losses += _confidence / self.config.n_layer
        
        
        # print("total loss ", losses)
        if targets is not None and (losses.item() < 0.05 or losses.isnan().any()):
            # print("oh no")
            pass
            # print("SAVED CONF ", savedConf)
            # # print("SAVED XE FACTOR ", savedXeFactor)
            # print("NLL MAX ", _nll_max)
        if all_logits:
            return allLogits, losses, trueLoss, confLoss, targetLoss, early_stop_num, earlyStopLayerDict
        else:
            return _logits, losses, trueLoss, confLoss, targetLoss, early_stop_num, earlyStopLayerDict
            # if _block_loss.item() < THRESHOLD and ENABLE_LAYER_LOSS: # this is average across entire T and B
            #     if master_process and print_weights:
            #         bprint(f"\tShort circuit at layer {i} with block_loss {_block_loss.item()}")
            #     break
        # print(len(self.transformer.h))

        # forward final layer norm and classifier
        # x = self.transformer.ln_f(x) # also share the final layer norm
        # logits = self.lm_head(x) # (B, T, vocab_size)
        # if all_logits:
        #     allLogits.append(logits)
        
        # if print_weights:
        #     # print(self.transformer.sharedblock.attn.c_attn.weight.view(-1)[-6:].data)
        #     probs = F.softmax(logits, dim=-1) # (B, T, vocab_size?)
        #     # do top-k sampling of 50 (huggingface pipeline default)
        #     # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        #     topk_probs, topk_indices = torch.topk(probs[0,-1,:], 5, dim=-1)
        #     print(f"Layer 6\t\t\t {topk_probs.tolist(), enc.decode(topk_indices.tolist())}")

        # trueLoss = None
        # if targets is not None:
        #     _logits = self.lm_head.requires_grad_(True)(x) # (B, T, vocab_size) # NOTE: this is with gradient
        #     # shape of input to x-entropy is B*T x V, B*T x 1
        #     # recall: logits are basically log counts
        #     # softmax = logits.exp (counts) / logits.exp (counts).sum(dim=-1, keepdim=True), i.e. normalized counts
        #     # cross entropy loss is just -log(pr[target])
        #     # F.cross_entropy takes average over B*T
        #     xe = F.cross_entropy(_logits.view(-1, _logits.size(-1)), targets.view(-1))
        #     # print("LOSS", xe.item())
        #     losses += xe
        #     trueLoss = xe.detach()

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
                f.write(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters\n")
                f.write(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters\n")
    

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


# torch.manual_seed(1234)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(1234)

# ===================
# HYPERPARAMETERS
# ===================

ALL_LAYER_LOSS = False
ELEMENTWISEAFFINE = False # whether LN parameters are learned


test_name="10-resmlp-single-axm-recurse"
test_description=f" Reusing blocks, max LR 6e-4, alllayerloss={ALL_LAYER_LOSS}, y = self.attn(y)*self.mlp(y), x=res+y, ELEMENTWISEAFFINE={ELEMENTWISEAFFINE}"

# Create log and persistence directory
log_dir = "log-ben"
sample_dir = "samples-ben"
checkpoint_dir = "checkpoints-ben"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{test_name}-log.txt")
sample_file = os.path.join(sample_dir, f"{test_name}-main.txt")
if master_process:
    with open(log_file, "w") as f: # clear log file
        pass
    with open(sample_file, "w") as f: # clear samples file
        pass

def bprint(s):
    print(s)
    with open(log_file, "a") as f:
        f.write(s + "\n")

# We want a larger batch size to follow GPT-3 Small, roughly B*T = 0.5M; but setting B = 488 will blow up the GPU.
# Since we only have small GPUs, we'll just simulate large batches using accumulation.
B = 8 # micro batch size, will do forward backward but not do an update yet # previously 16 # A100 can do 64?
T = 1024 # sequence length # 16 # 1024
total_batch_size = 8 * 16 * T # 524288 # B*T # TODO change to 524288 # 2**19 ~0.5M in number of tokens #32 
max_steps = 300000 + 1 # How many steps do we train for
# Implement cosine lr decay in the style of GPT-3
max_lr = 6e-4 # from GPT 3 paper # double it because it seems to work # quadruple it
min_lr = max_lr * 0.1
warmup_steps = 10
use_compile = False # May run into bugs
THRESHOLD = 0.1 # 0.1 # ln(0.9), about 90% confidence
ENABLE_LAYER_LOSS = False

hello_swag_frequency = 600000
validation_frequency = 2000
checkpoint_frequency = 20000
sample_frequency = 150

assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*(# gpus)"
grad_accum_steps = total_batch_size // (B*T * ddp_world_size) # 4; so each batch split into 4 mini batches
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"Mini-batch size: {B}*{T}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print(f"Training max steps: {max_steps}")
    print(f"Num GPUs: {ddp_world_size}")
    bprint(f"Threshold: {THRESHOLD}")
    bprint(f"Enable layer loss: {ALL_LAYER_LOSS}")
    bprint(f"MAX LEARNING RATE: {max_lr}")
    bprint(f"Experiment name: {test_name}")
    bprint(f"Experiment description: {test_description}")


    with open(log_file, "a") as f:
        f.write(f"total desired batch size: {total_batch_size}\n")
        f.write(f"Mini-batch size: {B}*{T}\n")
        f.write(f"=> calculated gradient accumulation steps: {grad_accum_steps}\n")
        f.write(f"=> calculated gradient accumulation steps: {grad_accum_steps}\n")
        f.write(f"Training max steps: {max_steps}")
        f.write(f"Num GPUs: {ddp_world_size}")

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
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
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
                    logits, _, loss, _, _, _, _= model(x, y)
                loss = loss / val_loss_steps # average accumulated loss
                val_loss_accum += loss.detach() # why do i need to call detach if I never call backward on it
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"@ {step} val {val_loss_accum.item():.4f}\n")
    
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
                    logits, _, loss, _, _, _, _ = model(tokens)
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
        max_length = 30 # 9
        # tokens = enc.encode("Hello, I'm a language model,")
        tokens = enc.encode("A Poem for you! Roses are red, Potatoes are ")
        XLEN = len(tokens)
        printgen = tokens
        leftParens = enc.encode("\t\t(")
        rightParens = enc.encode(")\n")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # different seed for each GPU
        while xgen.size(1) < max_length:
            if xgen.size(1) < XLEN + 1:
                _print_weights_val = True
            else:
                _print_weights_val = False
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _, _, _, _, _, _ = model(xgen[:,-T:], all_logits=True, print_weights=_print_weights_val) # (B, T, vocab_size)

                if xgen.size(1) < XLEN + 2:

                    printgen.extend(enc.encode("\n------\n"))    
                    for l in logits:
                        # take the last token logits
                        printgen.extend(leftParens)
                        for j in range(7, -1, -1): # 0 to 7
                            _logit = l[:,-(j+1),:] #(B, vocab_size?)
                            _probs = F.softmax(_logit, dim=-1) #(B, vocab_size?)
                            vals, idxs = _probs.max(dim=-1, keepdim=True) #(B, 1)
                            printgen.append(idxs[0,0].item())
                            printgen.extend(enc.encode(f":{vals[0,0].item():.4f}"))
                        printgen.extend(rightParens)

                logits = logits[-1] # (1, 8, 50304)
                # take logits at the last layer
                logits = logits[:,-1,:] # (1, 8, 50304)
                # get the probabilities
                probs = F.softmax(logits, dim=-1) # (1, 8, 50304)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 1, dim=-1) # TODO change 1 back to 50
                # print(topk_probs.shape) # (1, 50)
                # print(topk_indices.shape) # (1, 50)

                if xgen.size(1) < 10:
                    bprint(f"Final sample ({xgen.size(1) - 8} word) \n\t\t\t {topk_probs[0].tolist(), enc.decode(topk_indices[0].tolist())}")
                    # TODO figure out why this is different than the block_loss printing...


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
            bprint(f"rank {ddp_rank} sample {i}: {decoded}")
            with open(sample_file, "a") as f:
                f.write(f"{step}: sample {i}: {decoded}\n")
        
    # Actual training loop
    model.train()
    optimizer.zero_grad() # recall that .backwards() adds to gradients in pytorch, so must start at 0
    loss_accum = 0.0
    conf_loss_accum = 0.0
    target_loss_accum = 0.0
    loss_accum_all = 0.0
    earlyStopNum_accum = 0.0
    earlyStopLayerDict_accum = torch.zeros(12, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        timesBatchUsed = train_loader.num_times_latest_batch_used()
        x, y = x.to(device), y.to(device)
        if ddp:
            # (Kaparthy says: hope this isn't a breaking torch change, should maybe use no_sync)
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, trueloss, confLoss, targetLoss, earlyStopNum, earlyStopLayerDict = model(x, y) # NOTE: we don't actually use the logits
        # Need to scale loss by grad_accum_steps because we need to get the average loss over the entire batch (not just over mini batches)
        loss = loss / grad_accum_steps
        loss_accum += trueloss.detach() / grad_accum_steps
        loss_accum_all += loss.detach()
        earlyStopNum_accum += earlyStopNum
        conf_loss_accum += confLoss.detach() / grad_accum_steps
        target_loss_accum += targetLoss.detach() / grad_accum_steps
        earlyStopLayerDict_accum += earlyStopLayerDict.detach()
        # .backward() will just += the gradient
        # DDP will automatically allreduce this for us
        loss.backward()
        # NOTE: I want each backward call to accumulate gradient on a different layer of the network... so I can do layerwise training
        # Each time I do a forward pass, i want it to remember the gradient for most of the backward calls, but not the very next backward call...
        # When summing losses, maybe i should just use a detached version
    
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

        earlyStopLayerDict_accum /= (B*T*grad_accum_steps)

        rList = [round(value, 2) for value in earlyStopLayerDict_accum.tolist()]

        bprint(f"@ {step} train {loss_accum.item():.4f} , allloss: {loss_accum_all.item():.4f}, confloss: {conf_loss_accum.item():.4f}, targetloss: {target_loss_accum.item():.4f}, earlystop: {earlyStopNum_accum / (B*T*grad_accum_steps):.3f}, earlystopdict: {rList}, lr:{lr:.4e}, norm:{norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}, flops:{flops / 1e12:.2f}, batch-reuse:{timesBatchUsed}") 


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
