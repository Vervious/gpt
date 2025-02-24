
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time
import hellaswag

import torch.autograd as autograd

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


class SigmoidAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size),diagonal=-1).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, z):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.sigmoid(att) + torch.eye(T, device=x.device) #broadcasts, .view(1, 1, T, T) # (B, nh, T, T)
        y = att @ z.unsqueeze(1) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, n_embd)
        y = y.sum(dim=1) # sum up the head dimension
        return y

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
        # key, query batched for all heads
        self.c_attn = nn.Linear(config.n_embd, 2*config.n_embd, bias=False)
        # value (normally, we should batch) self.c_attn_value = nn.Linear(config.n_embd, 3*config.n_embd, bias=False)
        self.c_attn_value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # if LOW_RANK_ATTN:
        #     # Deepseek style low-rank compression of kv cache, see deepseek v2 tech report
        #     self.d_proj = nn.Linear(config.n_embd, 4*config.n_embd / config.n_head)
        #     # TODO finish implementing

        # self.c_attn.ATTN_INIT = 1
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # a flag for scaling initialization to compensate for increase in variance due to residual connections (variance grows if I keep summing)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.cachedResW = None
        # self.cachedY = None

        # Don't need this once we switched to flashattention
        #  # not really a 'bias', more of a mask, following OpenAI naming
        # self.register_buffer("bias", torch.tril(
        #     torch.ones(config.block_size, config.block_size))
        #     .view(1,1,config.block_size,config.block_size)
        # )
        # no zeros
# @flag attention mask [ATTENTION_MASK]
        if ATTENTION_MASK:
            if ATTENTION_SINK:
                T = config.block_size + 1
            else:
                T = config.block_size
            mask = torch.triu(torch.ones(T, T),diagonal=1)
            mask = torch.where(mask == 1, float("-inf"), torch.tensor(0.0))
            if ATTENTION_SINK:
                mask[:,0] = 2*(torch.arange(0, T) + 1).log()
            mask = mask.view(1, 1, T, T)
            self.register_buffer("mask", mask)
# @endflag attention mask
        # print(torch.diagonal(self.mask, dim1=-2, dim2=-1).sum().item())

        if DELETE_SELF_CONTRIBUTION or EXTRACT_SELF_CONTRIBUTION:
            T = config.block_size
            self.register_buffer("nodiagonal", (1 - torch.eye(T, T)).view(1, 1, T, T))
        #     # ^ 0s on diagonal, 1 everywhere else


    def forward(self, x, z=None,print_weights=False, kvCache=None):
        # x used to compute Q, K; z replaces v if VALUE_MATRIX = False
        # z must be same dim as x
        B, T, C = x.size()

        if ATTENTION_SINK:
            # (1, 1, C)
            # add an all zeros token to x ("Zero Sink" from literature)
            s = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype) # (B, 1, C)

            x = torch.cat((s, x), dim=1) 
            # ^ (B, T, C) -> (B, T+1, C)

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", C is number of channels is nh * hs
        # e.g. GPT2 (124M), n_head = 12, hs = 64, nh*hs=C=768 channels

        T_with_cache = T
        newKvCache = None
        # each token emits three vectors query, key, value
        if VALUE_MATRIX or z is None:
            kq = self.c_attn(x)  # (B, T, C) -> (B, T, 2*C)
            v = self.c_attn_value(x)
            k,q = kq.split(self.n_embd, dim=2)


            if kvCache is not None:
                # note that in this case, we should not have fed in old x along the T dimension
                kOld, vOld = kvCache
                
                k = torch.cat((kOld, k), dim=1) # save along the T dimension
                v = torch.cat((vOld, v), dim=1) # save along the T dimension
                T_with_cache = k.size(1)

                newKvCache = (k, v)

            v = v.view(B, T_with_cache, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            # NOTE: don't need the below because we are not calculating ResW
            if ATTENTION_SINK:
                # NOTE: commandeer the last dimension to hold our data (the network
                # better not store anything useful there)
                extra_zeros = torch.ones(C // self.n_head, device=x.device, dtype=x.dtype)
                extra_zeros[-1] = 0
                # v_padded = torch.cat((v, extra_zeros), dim=-1) # (B, nh, T, hs+1)
                v = v * extra_zeros # zero out the last dimension (broadcast)
                v[:, :, 0, :] = 0 # zero out first token everywhere
                v[:, :, 0, -1] = 1
            
                
                
            if MEASURE_SELF_CONTRIBUTION or DELETE_SELF_CONTRIBUTION or EXTRACT_SELF_CONTRIBUTION:
                v2 = v
                v = torch.eye(T, device=x.device).view(1, 1, T, T) # (B, nh, T, T)
                # TODO a faster way to get the diagonal?
      
        else:
            assert False, "not yet implemented"
            qk = self.c_attn(x)
            q,k = qk.split(self.n_embd, dim=2)
            v = torch.eye(T, device=x.device).view(1, 1, T, T) # (B, nh, T, T)
            # v = z.unsqueeze(1) # (B, 1, T, C)

        # if print_weights:
        #     bprint(f"Kraw: {k[-1, -1, :10]}")
        #     bprint(f"Qraw: {q[-1, -1, :10]}")
        # treat heads as batches
        k = k.view(B, T_with_cache, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T_with_cache, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


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
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  
        # if print_weights:
        #     bprint(f"dtype {self.mask.dtype}")
        #     # bprint(f"my mask: {self.mask[:,:,:T,:T]}")
        if ATTENTION_MASK:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask[:,:,:T,:T])
        else:
            # k is (B, nh, cacheT, hs)
            # q is (B, nh, T, hs) # maybe we should disable token-level self attention? unclear
            # v is (B, nh, cacheT, hs)
            # q @ k.transpose -> T x hs @ hs x cacheT -> T x cacheT; i.e. scores for each destination token T (usually just one) for each source token cacheT
            # y = att @ v -> T x cacheT @ cacheT x hs -> T x hs, a linear combination
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        if print_weights:
            pass # fix later
            # with torch.no_grad():
            #     vv = torch.eye(T, device=x.device).view(1, 1, T, T)

            #     if ATTENTION_MASK:
            #         yy = F.scaled_dot_product_attention(q.detach(), k.detach(), vv, attn_mask=self.mask[:,:,:T,:T])
            #     else:
            #         yy = F.scaled_dot_product_attention(q.detach(), k.detach(), vv, is_causal=True)
                    
            #     torch.set_printoptions(linewidth=300, sci_mode=False)
            #     bprint(f"Attn {T} {yy[-1,-1,:,:]}") #TODO uncomment
            #     # bprint(f"Tied Weights? {T} {self.c_attn.weight}")

            #     rawATT = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
            #     # bprint(f"Kweights\n{self.c_attn.weight[:self.n_embd, :]}")
            #     # bprint(f"K: {k[-1, -1, -1, :]}")
            #     # bprint(f"Qweights\n{self.c_attn.weight[self.n_embd:2*self.n_embd, :]}")
            #     # bprint(f"Q: {q[-1, -1, -1, :]}")
            #     # bprint(f"RAWVALUES (nomask)\n{rawATT[-1,-1,:,:]}")
            #     if ATTENTION_MASK:
            #         bprint(f"RAWVALUES (withmask)\n{rawATT[-1,-1,:,:] + self.mask[:,:,:T,:T]}")
            #     if self.cachedResW is not None:
            #         torch.set_printoptions(linewidth=200, sci_mode=True, threshold=float('inf'))
            #         bprint(f"GRAD\n{self.cachedResW.grad[-1, -8:, :]}")
            #     # bprint(f"========")

        scores = None

        if VALUE_MATRIX or z is None:
            # recall y is (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble all head outputs side by side

            # (B, T, n_embd)
            # output projection
            y = self.c_proj(y) # NOTE: what is the point of this (to support dimension reduction from before, i don't think we actualy need to do dimension reduction)

        else:
            assert False, "not yet implemented"
            if ATTENTION_SINK:
                pass

            # TODO ATTENTION SINK NOT SUPPORTED YET
            if MEASURE_SELF_CONTRIBUTION:
                # without value matrix: (B, nh, T, T)
                scores = torch.diagonal(y, dim1=-2, dim2=-1).detach() # (B, nh, T)
                scores = scores.sum(dim=1).unsqueeze(-1) # (B, T, 1)
            if EXTRACT_SELF_CONTRIBUTION:
                resx = torch.diagonal(y, dim1=-2, dim2=-1).unsqueeze(-1) * z.unsqueeze(1) # (B, nh, T, 1) * (B, 1, T, C)
                resx = resx.sum(dim=1) / self.n_head
            if DELETE_SELF_CONTRIBUTION or EXTRACT_SELF_CONTRIBUTION:
                y = y*self.nodiagonal[:,:,:T,:T] # delete the self contribution
            y = y @ z.unsqueeze(1) # (B, nh, T, T) @ (B, 1, T, C) -> (B, nh, T, C)
            y = y.sum(dim=1) / self.n_head # sum up the head dimension

        # TODO deal with EXTRACT_SELF_CONTRIBUTION and delete, remove it if unused
        if ATTENTION_SINK:
            assert False, "not yet implemented"
            if scores is not None:
                scores = scores[:, 1:, :]
            # resw[:,1:,:]
            return y[:, 1:, :], scores # remove the first token (average token)
        else:
            # return the new kvCache
            return y, newKvCache


class Gate(DualModule):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = x.sum(dim=-1, keepdim=True) # sum over the last dimension
        x = torch.sigmoid(x)
        return x
    
class MLPFewParams(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.inputWidth = config.n_embd
        self.hiddenWidth = MLP_HIDDENWIDTH_INTERPRETER
        self.gelu = nn.GELU(approximate='tanh') # consider using somehting else

        # TODO: should inner matrix really be square?
        self.fU = nn.Parameter(torch.empty(MLPMAT_INNER_SIZE, self.inputWidth))
        self.fV = nn.Parameter(torch.empty(self.hiddenWidth, MLPMAT_INNER_SIZE))

        self.bU = nn.Parameter(torch.empty(MLPMAT_INNER_SIZE, self.hiddenWidth))
        self.bV = nn.Parameter(torch.empty(self.inputWidth, MLPMAT_INNER_SIZE))

        torch.nn.init.normal_(self.fU, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.fV, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.bU, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.bV, mean=0.0, std=0.02)


    def forward(self, x, front, back, hiddenBias):

        B, T, paramSize = front.size()
        B2, T2, paramSize2 = back.size()
        B3, T3, C = x.size()
        B4, T4, hiddenWidth = hiddenBias.size()

        assert hiddenWidth == self.hiddenWidth, f"pMLPMAT wrong hidden width {hiddenWidth} {self.hiddenWidth}"

        assert B == B2 and T == T2 and paramSize == paramSize2, f"Parametrized MLP misconfig {B} {T} {paramSize} {B2} {T2} {paramSize2}"
        assert B == B3 and T == T3, f"pMLPMAT wrong x size {B} {T} {B3} {T3}"
        assert MLPMAT_INNER_SIZE**2 == paramSize, f"pMLPMAT misconfig {MLPMAT_INNER_SIZE} {paramSize}"

        compressedFront = front.view(B, T, MLPMAT_INNER_SIZE, MLPMAT_INNER_SIZE)
        compressedBack = back.view(B, T, MLPMAT_INNER_SIZE, MLPMAT_INNER_SIZE)

        x = x.unsqueeze(-1) # (B, T, C, 1)
        # apply the forward MLP
        x = (self.fV @ (compressedFront @ (self.fU @ x))) # (B, T, hidden_width, 1)
        x = x + hiddenBias.unsqueeze(-1)
        # apply gelu
        x = self.gelu(x)
        # apply the backward MLP
        x = (self.bV @ (compressedBack @ (self.bU @ x))) # (B, T, inputWidth, 1)
        x = x.squeeze(-1)

        # NOTE: last layer probably doesn't need a bias, see `14-nooutlinearbiasmlp` experiment
        return x


class BenCompiler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_SCALE * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, MLP_HIDDENWIDTH_INTERPRETER + 2*MLPMAT_INNER_SIZE*MLPMAT_INNER_SIZE)

        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        hiddenBias, fParams, bParams = x.split([MLP_HIDDENWIDTH_INTERPRETER, MLPMAT_INNER_SIZE*MLPMAT_INNER_SIZE, MLPMAT_INNER_SIZE*MLPMAT_INNER_SIZE], dim=-1)
        return hiddenBias, fParams, bParams
    

class BenCompiler2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_SCALE * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, config.n_embd + MLPMAT_INNER_SIZE*MLPMAT_INNER_SIZE)

        # self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        hiddenBias, fParams = x.split([self.n_embd, MLPMAT_INNER_SIZE*MLPMAT_INNER_SIZE], dim=-1)
        return {"bias": hiddenBias, "fparams": fParams}


class MLPMatApply(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.U = nn.Parameter(torch.empty(config.n_embd, MLPMAT_INNER_SIZE))
        self.V = nn.Parameter(torch.empty(MLPMAT_INNER_SIZE, config.n_embd))
        
        torch.nn.init.normal_(self.U, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.V, mean=0.0, std=0.02)
    
    def forward(self, program, attn):
        # m has dimension (B, T, 3*C)
        m = program["fparams"]
        bias = program["bias"]
        B, T, matnumparams = m.size()

        assert MLPMAT_INNER_SIZE**2 == matnumparams, f"MLPMAT misconfig {MLPMAT_INNER_SIZE} {matnumparams}"
        m = m.view(B, T, MLPMAT_INNER_SIZE, MLPMAT_INNER_SIZE)

        attn = attn.unsqueeze(-1) # (B, T, C, 1)

        # self.U @ m --> (C, 48) @ (B, T, 48, 48) = (B, T, C, 48)
        # Above @ V --> (B, T, C, 48) @ (48, C) = (B, T, C, C)
        # Above @ attn --> (B, T, C, C) @ (B, T, C, 1) = (B, T, C, 1)
        return (self.U @ (m @ (self.V @ attn))).squeeze(-1) + bias


class BenCompilerNoOp(nn.Module):

    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        return x

class MLPConcat(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd * 2, MLP_SCALE * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.ln = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)

    def forward(self, x, attn):
        x = self.ln(x)
        z = torch.cat((x, attn), dim=-1) # (B, T, 2C)
        x = self.c_fc(z)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# @flag machine_code [UNSET]
class MultExecute(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(config)

    def forward(self, program, attn):
        return self.mlp(program) * attn
# @endflag machine_code


class BenExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLPConcat(config)
    def forward(self, program, attn):
        return self.mlp(program, attn)


class VanillaExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_SCALE * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)

    def forward(self, x, attn):
        inp = self.ln_2(x + attn)
        y = self.c_fc(inp)
        y = self.gelu(y)
        mlp = self.c_proj(y)
        return mlp + attn

class NoAttnExecute(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_SCALE * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
    def forward(self, x, attn):
        inp = self.ln_2(x)
        y = self.c_fc(inp)
        y = self.gelu(y)
        mlp = self.c_proj(y)
        return mlp

class BenBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
        self.attn = CausalSelfAttention(config)
        self.n_head = config.n_head
        self.n_layer = config.n_layer
# @flag machine_modules
        self.compiler = BenCompilerNoOp(config)
        self.execute = VanillaExecute(config)
# @endflag machine_modules

        # self.throughput = nn.Parameter(torch.tensor(-2.0))
        # torch.nn.init.normal_(self.throughput, mean=-2.0, std=0.02)

        

        # self.mlp = MLP(config)

    def forward(self, x, kvCache, print_weights=False,step=0):
        # x should be of size (B, 1, n_embd)
        # kvCache should be of size (B, T, C), (B, T, C); here C is also n_embd as attention is configured
        metadata = {}

        if step > 2 and step < self.n_layer - 2:
            print_weights = False

        # stronger attention should result in machineOutput with larger norm,
        # which should then get normalized; then learning to output a larger norm
        # vs a smaller norm is our mechanism for gating attention

        # First LN important to make sure the signal to attn does not get too small.
        # cannot LN the output, why? 

# @flag block_logic
        y = self.ln_1(x)
        attn, newKvCache = self.attn(y, y, print_weights=print_weights, kvCache=kvCache)
        program = self.compiler(x)
        machineOutput = self.execute(program, attn)
        newx = x + machineOutput
# @endflag block_logic


        metadata["_norm_attn"] = attn.std(dim=-1).mean() / self.n_layer #torch.linalg.norm(attn, dim=-1).mean().item()
        metadata["_norm_y"] = y.std(dim=-1).mean() / self.n_layer # should be 1 / 12
        metadata["_norm_x"] = x.std(dim=-1).mean() / self.n_layer
        metadata["_norm_output"] = machineOutput.std(dim=-1).mean() / self.n_layer
        # metadata["_frac_noop"] = xWeights.mean() / self.n_layer

        x = newx

        # if scores is not None:
        #     a = scores.detach()
        #     metadata["zero"] = (a < 5).sum().item()
        #     metadata["neg"] = (a < 0.5).sum().item()
        #     metadata["pos"] = (a > 5).sum().item()

        return x, metadata, newKvCache  

class ApplyMatrixFromFewParams(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.U = nn.Parameter(torch.empty(config.n_embd, MLPMAT_INNER_SIZE))
        self.V = nn.Parameter(torch.empty(MLPMAT_INNER_SIZE, config.n_embd))
        
        torch.nn.init.normal_(self.U, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.V, mean=0.0, std=0.02)
    
    def forward(self, m, attn):
        # m has dimension (B, T, 3*C)
        B, T, matnumparams = m.size()

        assert MLPMAT_INNER_SIZE**2 == matnumparams, f"MLPMAT misconfig {MLPMAT_INNER_SIZE} {matnumparams}"
        m = m.view(B, T, MLPMAT_INNER_SIZE, MLPMAT_INNER_SIZE)

        attn = attn.unsqueeze(-1) # (B, T, C, 1)

        # self.U @ m --> (C, 48) @ (B, T, 48, 48) = (B, T, C, 48)
        # Above @ V --> (B, T, C, 48) @ (48, C) = (B, T, C, C)
        # Above @ attn --> (B, T, C, C) @ (B, T, C, 1) = (B, T, C, 1)
        return (self.U @ (m @ (self.V @ attn))).squeeze(-1)

class MLP(DualModule):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_SCALE * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, config.n_embd, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class FatMLP(DualModule):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_SCALE * config.n_embd)
        # smoother ReLU, also maybe helps dead neurons empirically?
        # (should just delete dead neurons)
        self.gelu = nn.GELU(approximate='tanh')  # should just use 'none' if not trying to copy GPT2
        self.c_proj = nn.Linear(MLP_SCALE * config.n_embd, config.n_embd + MATRIX_NUM_PARAMS)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        b, m = x.split([self.n_embd, MATRIX_NUM_PARAMS], dim=-1)
        return m, b



class VanillaBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
        self.mlp = MLP(config)
        self.n_head = config.n_head

    def forward(self, x,print_weights=False):
        # # VANILLA
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))

        # # AXM
        # y = self.ln_1(x)
        # x = x + self.attn(y)*self.mlp(y)

        # y = self.ln_1(x)
        # attn = self.attn(y)
        # m, bias = self.fatmlp(y) # (B, T, 3*C), (B,T,C)
        # M = self.applymat(m, attn) #(B, T, 3*C), (B, T, C) -> (B, T, C)
        # x = M + bias + x
        metadata = {}

        # y = self.ln_1(x)
        # attn = self.attn(y,y)
        # siz = torch.linalg.vecdot(y, y,dim=-1).unsqueeze(-1) # (B, T, 1)
        # app = torch.linalg.vecdot(attn, y,dim=-1).unsqueeze(-1) / siz # may be greater than 1
        # app = (torch.sigmoid(torch.abs(app)) - 0.5) * 2  # [0, 1]
        # m, bias = self.fatmlp(y)
        # M = self.applymat(m, attn) #(B, T, 3*C), (B, T, C) -> (B, T, C)
        # x = M + bias + x

        # scores is (B, T, 1)

        # y = self.ln_1(x)
        # attn, scores = self.attn(y, y)
        # x = x + self.mlp(attn)
        y = self.ln_1(x)
        attn, scores = self.attn(y, y,print_weights=print_weights)
        x = x + attn
        x = y + self.mlp(self.ln_2(x))

        # print(scores[-1,-1,-1])

        if scores is not None:
            a = scores.detach()
            metadata["zero"] = (a < 5).sum().item()
            metadata["neg"] = (a < 0.5).sum().item()
            metadata["pos"] = (a > 5).sum().item()

        return x, metadata
    
class Block(DualModule):

    def __init__(self, config):
        super().__init__()
        self.ln = nn.RMSNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE)
        self.attn = CausalSelfAttention(config)
        self.gate = Gate(config)
        self.mlp = MLP(config)
        # self.rotator = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self, x, res, y):
        # note residual connection res
        # out = self.ln_1(x)
        # NOTE, for some reason LN(x) + self.attn(LN(x)) doesn't work as well. 
        # ^ it is incredibly important that the residual is not passed through the layer norm... (TODO why??? Layers can no-op?)
        # x = x + self.attn(self.ln_1(x))  # reduce operation (all to all)
        # NOTE: res will generally be very large...
        # mlp2 = self.mlp2(x)
        attn = self.attn(x, x)
        mlp = self.mlp(x)
        midx = mlp
        y = mlp*res
        x = y + res + attn
        newres = x
        x = self.ln(x)
        # res + self.attn(x) # NOTE that the residual connection x is already layer normed, unlike usual transformer implementation # TODO add back residual res + . NOTE x + self.attn(x) is simply horrible (why?)... we cannot layer norm it (prev too big or too small?)
        # Maybe the layernorm just destroys relative magnitude of things...
        # NOTE, likewise LN(x) + mlp(LN(x)) doesn't work as well? The residual literally has to be untouched. 
        # midx = y # TODO UNCOMMENT, and then try removing res (applying attn to res)
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
        return x, newres, midx, y, attn, mlp, res


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
        self.VANILLA = True

        sharedBlock = Block(config)

        if CODE_MODE:
            self.T2 = 128
            self.code = nn.Parameter(torch.zeros(self.T2, config.n_embd, dtype=torch.float32))
        else:
            self.T2 = 0

        if self.VANILLA:
            if REUSE_WEIGHTS:
                sharedBlock = BenBlock(config)

                self.transformer = nn.ModuleDict(dict(
                    wte = nn.Embedding(config.vocab_size, config.n_embd),
                    wpe = nn.Embedding(config.block_size + self.T2, config.n_embd),
                    sharedblock = sharedBlock,
                    ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE),
                ))
            else:
                self.transformer = nn.ModuleDict(dict(
                    wte = nn.Embedding(config.vocab_size, config.n_embd),
                    wpe = nn.Embedding(config.block_size + self.T2, config.n_embd),
                    h = nn.ModuleList([BenBlock(config) for _ in range(config.n_layer)]),
                    ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=ELEMENTWISEAFFINE),
                ))

# @flag attn_weights [TIE_ATTN_WEIGHTS]
                if TIE_ATTN_WEIGHTS:
                    # Tie model weights together
                    firstBlock = self.transformer.h[0]
                    for block in self.transformer.h:
                        block.attn.c_attn.weight = firstBlock.attn.c_attn.weight
                        # block.attn = firstBlock.attn
# @endflag attn_weights

# @flag mlp_weights [TIE_MLP_WEIGHTS]
                if TIE_MLP_WEIGHTS:
                    # Only works with BenBlock, set module to be the same
                    firstBlock = self.transformer.h[0]
                    for block in self.transformer.h:
                        block.execute = firstBlock.execute
# @endflag mlp_weights
             
        else:

            self.transformer = nn.ModuleDict(dict(
                # weights of token embedding
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                # weights of position embedding
                wpe = nn.Embedding(config.block_size, config.n_embd),
                # h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #  COMMEZNT
                sharedblock = sharedBlock, # NOTE: this does not seem to degrade performance at least early in the training process
                # weights of layer normalization
                ln_f = sharedBlock.ln, # nn.LayerNorm(config.n_embd),
                # NOTE we share ALL layer norms which may not be necessarily wise
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # copies the data pointer


        # self.router = nn.Parameter(torch.zeros(config.n_layer, config.n_layer, dtype=torch.float32))

        # param initialization
        self.apply(self._init_weights) # apply iterates all submodules

# @flag fixed_attn [NO_GRAD_ATTN]
        if NO_GRAD_ATTN:
            for block in self.transformer.h:
                block.attn.c_attn.weight.requires_grad = False
# @endflag fixed_attn
    
    def _init_weights(self, module):
        # No need to change default initialization of LayerNorm
        if isinstance(module, nn.Linear):
            stdConfig = 0.02
            if hasattr(module, 'ATTN_SCALE_INIT'):
                #module.weight shoudl have dimension n_embd * 3n_embd
                # this should make res 1
                N = self.config.n_embd
                torch.nn.init.normal_(module.weight[:N,:], mean=-1.0, std=stdConfig)
                with torch.no_grad():
                    module.weight[N:2*N,:] = -1 * module.weight[:N,:]
                torch.nn.init.normal_(module.weight[2*N:,:], mean=0.0, std=stdConfig)
            elif hasattr(module, 'ATTN_INIT'):
                head_size = self.config.n_embd // self.config.n_head
                STD_DIM = math.sqrt(1.0 / head_size)
                torch.nn.init.normal_(module.weight, mean=0.0, std=STD_DIM) # init to random matrix (todo consider Rademacher matrix or something sparser)
            else:
                if hasattr(module, 'NANOGPT_SCALE_INIT'):
                    # 2x because we have two linears per layer: block.attn and block.mlp
                    stdConfig *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=stdConfig)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
# @flag init_logic [CODE_MODE]
        elif isinstance(module, GPT) and CODE_MODE:
            torch.nn.init.normal_(module.code, mean=0.0, std=5)
# @endflag init_logic

    def vanillaforward(self, idx, targets=None,print_weights=False):
        device = idx.device

        # in sequential world, we want b parallel threads of thought, of length t,
        # predicting up to t+1th token
        # this design will really neuter the parallel trainability of our model

        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)


        pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        loss = torch.tensor(0.0, device=idx.device)
        trueloss = None

        logits = []

        outerMetadata = {}

        # here, x is (B, T, n_embd)
        # we want to iterate through it one at a time, (B, 1, n_embd)

        kvCache = None
        xj = None
        for j in range(T):
            xj = x[:,j:j+1,:] # (B, 1, n_embd)
            # if kvCache:
            #     # we do not want to backprop too deep...
            #     # for embeddings belonging to previous j, they should only
            #     # be influenced in so far as predicting j+1.
            #     kvCache = kvCache.detach()

            for i in range(self.config.n_layer):
                if REUSE_WEIGHTS:
                    block = self.transformer.sharedblock
                else:
                    block = self.transformer.h[i]
                
                # kvCache is size (B, j*n_layer + i, C), where C is currently set to n_embd
                xj, metadata, newKvCache = block(xj, kvCache, print_weights=print_weights,step=i)
                kvCache = newKvCache

                for key, value in metadata.items():
                    outerMetadata[key] = outerMetadata.get(key, 0) + value
            
            if targets is not None:
                xj = self.transformer.ln_f(xj)
                _logits = self.lm_head(xj)
                logits.append(_logits)

                # We want to compare with a (B, 1, vocab_size) target
                targetj = targets[:,j] # Note that targets contains indices, so is actually shape (B, T)
                trueloss = F.cross_entropy(_logits.view(-1, _logits.size(-1)), targetj.view(-1), ignore_index=-1)
                loss = trueloss
            else:
                # compute a loss using the next token anyways during inference? TODO
                pass

        if targets is None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # NOTE: during inference, instead of inferring from the total, infer only from the last layer
            # _logits used to be (B, T, vocab_size). Now, it is (B, 1, vocab_size)
            _logits = self.lm_head(self.transformer.ln_f(xj))
            logits.append(_logits)
            loss = None
        
        return logits, loss, trueloss, outerMetadata

    def forward(self, idx, targets=None, all_logits=False, print_weights=False, dump_val=False):
        # this chunk of code is a remnant from past hackery
        zero = torch.tensor(0.0, device=idx.device)
        earlyStopLayerDict = torch.zeros(self.config.n_layer, device=idx.device)

        vanillalogits, vanillaloss, trueloss, metadata = self.vanillaforward(idx, targets,print_weights=print_weights)

        if all_logits:
            return vanillalogits, vanillaloss, trueloss, zero, zero, metadata, earlyStopLayerDict
        else:
            return vanillalogits[-1], vanillaloss, trueloss, zero, zero, metadata, earlyStopLayerDict


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
    print("YES DDP")
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
test_name="19-funexperiment"

# We want a larger batch size to follow GPT-3 Small, roughly B*T = 0.5M; but setting B = 488 will blow up the GPU.
# Since we only have small GPUs, we'll just simulate large batches using accumulation.
B = 1280 # micro batch size, will do forward backward but not do an update yet # previously 16 # A100 can do 64?
T = 128 # sequence length # 16 # 1024
config_ = GPTConfig(vocab_size=50304, block_size=T, n_layer=12)#, n_embd=1296) #, n_layer=24, n_head=16, n_embd=1024)


total_batch_size = B * T # 8 * 16 * T # 524288 # B*T # TODO change to 524288 # 2**19 ~0.5M in number of tokens #32 
max_steps = 300000 + 1 # How many steps do we train for
# Implement cosine lr decay in the style of GPT-3
max_lr = 6e-4 # from GPT 3 paper # double it because it seems to work # quadruple it
min_lr = max_lr * 0.025
warmup_steps = 100 # 100
use_compile = False # May run into bugs
THRESHOLD = 0.1 # 0.1 # ln(0.9), about 90% confidence
ENABLE_LAYER_LOSS = False


NEW_ALL_LAYER_LOSS = False
ELEMENTWISEAFFINE = True # whether LN parameters are learned
VALUE_MATRIX = True
MLP_SCALE = 4
MLP_HIDDENWIDTH_INTERPRETER = config_.n_embd
REUSE_WEIGHTS = False
MEASURE_SELF_CONTRIBUTION = False
DELETE_SELF_CONTRIBUTION = False
EXTRACT_SELF_CONTRIBUTION = False # TODO UNUSED
MLPMAT_INNER_SIZE = 64 # note 48^2 = 2304 = 3*768 = 3*n_embd
MATRIX_NUM_PARAMS = MLPMAT_INNER_SIZE*MLPMAT_INNER_SIZE # see prev line
ATTENTION_SINK = False
ATTENTION_MASK = False
IDENTITY_LOSS = False
CODE_MODE = False
TIE_ATTN_WEIGHTS = True
TIE_MLP_WEIGHTS = False
NO_GRAD_ATTN = False
LOW_RANK_ATTN = True

import observability
flag_str = observability.extract_flagged_code()

# Reusing blocks, max LR 6e-4, alllayerloss={ALL_LAYER_LOSS}, 
test_description=f"""```
Transformer, max LR {max_lr} n_layer {config_.n_layer}
Setting:
==details======
{flag_str}
========
VALUEMATRIX={VALUE_MATRIX}
REUSE_WEIGHTS={REUSE_WEIGHTS}
MLP_SCALE={MLP_SCALE}
ATTENTION_SINK={ATTENTION_SINK}
TIE_ATTN_WEIGHTS={TIE_ATTN_WEIGHTS}
LOW_RANK_ATTN={LOW_RANK_ATTN}
```
![caption](img/{test_name}.jpg)
"""


ALL_LAYER_LOSS = False

# Create log and persistence directory
log_dir = "log-ben"
superlog_dir = "superlog"
sample_dir = "samples-ben"
checkpoint_dir = "checkpoints-ben"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(superlog_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{test_name}-log.txt")
superlog_file = os.path.join(superlog_dir, f"{test_name}.txt")
sample_file = os.path.join(sample_dir, f"{test_name}-main.txt")
if master_process:
    with open(log_file, "w") as f: # clear log file
        pass
    with open(sample_file, "w") as f: # clear samples file
        pass
    with open(superlog_file, "w") as f: # clear samples file
        pass

def bprint(s):
    print(s)
    with open(log_file, "a") as f:
        f.write(s + "\n")

def cprint(s):
    # print(s)
    with open(superlog_file, "a") as f:
        f.write(s + "\n")

def cclear():
    print("Clearing contents of superlog")
    with open(superlog_file, "w") as f:
        # Clear the contents of the file
        pass


hello_swag_frequency = 600000
validation_frequency = 2000
checkpoint_frequency = 5000
sample_frequency = 250
inner_dump_frequency = 500

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
    bprint(f"MLPSCALE: {MLP_SCALE}")
    bprint(f"Experiment description: \n{test_description}")
    bprint(f"Warmup steps: {warmup_steps}")

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
bprint(str(config_.__dict__))
model = GPT(config_) # model = GPT.from_pretrained('gpt2')
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
    if (step > max_steps - 2 and (not use_compile)) or (step % sample_frequency == 50 and (not use_compile)) or step == 0 or step == 1: # > 0 and step % 100 == 0: # Like Kaparthy, I run into compilation issues
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
            _dump_val = False
            if xgen.size(1) < XLEN + 1:
                _print_weights_val = True
                if (step % inner_dump_frequency == 50):
                    cclear()
                    cprint(f"! {step}")
                    _dump_val = True

                # bprint(f"Code: {model.module.code}")
            else:
                _print_weights_val = False
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _, _, _, _, _, _ = model(xgen[:,-T:], all_logits=True, print_weights=_print_weights_val, dump_val=_dump_val) # (B, T, vocab_size)

                if xgen.size(1) < XLEN + 2:

                    if xgen.size(1) < XLEN + 1 and CODE_MODE:
                        _code = model.module.code
                        _code_interp = model.module.lm_head(_code)
                        # print(_code_interp.shape)
                        printgen.extend(enc.encode("\n>>>>>>\n"))   
                        printgen.extend(leftParens)
                        for j in range(7, -1, -1): # 0 to 7
                            _logit = _code_interp[-(j+1),:] #(1, vocab_size?)
                            _probs = F.softmax(_logit, dim=-1) #(vocab_size?)
                            # print(_logit)
                            vals, idxs = _probs.max(dim=-1, keepdim=True) #(B, 1)
                            printgen.append(idxs[0].item())
                            printgen.extend(enc.encode(f":{vals[0].item():.4f}"))
                        printgen.extend(rightParens) 

                    printgen.extend(enc.encode("\n------\n"))    
                    for l in logits:
                        # take the last token logits
                        printgen.extend(leftParens)
                        _logit = l #l should already be (B, 1, vocab_size?)
                        _probs = F.softmax(_logit, dim=-1) #(B, 1, vocab_size?)
                        vals, idxs = _probs.topk(7, dim=-1) #(B, 1, 7)
                        for j in range(0, 7):
                            # print top j
                            printgen.append(idxs[0,0,j].item())
                            printgen.extend(enc.encode(f":{vals[0,0,j].item():.4f}"))
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
    _outer_metadata = {}

    earlyStopLayerDict_accum = torch.zeros(config_.n_layer, device=device)
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        timesBatchUsed = train_loader.num_times_latest_batch_used()
        x, y = x.to(device), y.to(device)
        if ddp:
            # (Kaparthy says: hope this isn't a breaking torch change, should maybe use no_sync)
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        _print_weights_val = False
        if micro_step == grad_accum_steps - 1 and step == 1:
            _print_weights_val = True
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, trueloss, confLoss, targetLoss, metadata, earlyStopLayerDict = model(x, y, print_weights=_print_weights_val) # NOTE: we don't actually use the logits
        # Need to scale loss by grad_accum_steps because we need to get the average loss over the entire batch (not just over mini batches)
        loss = loss / grad_accum_steps
        loss_accum += trueloss.detach() / grad_accum_steps
        loss_accum_all += loss.detach()

        for k, v in metadata.items():
            _outer_metadata[k] = _outer_metadata.get(k, 0.0) + v

        conf_loss_accum += confLoss.detach() / grad_accum_steps
        target_loss_accum += targetLoss.detach() / grad_accum_steps
        earlyStopLayerDict_accum += earlyStopLayerDict.detach()
        # .backward() will just += the gradient
        # DDP will automatically allreduce this for us
        loss.backward()
        # print(f"GPT BACKED LMHEAD: {model.module.lm_head.weight.grad}")
        # print(f"GPT BACKED: {model.module.router.grad}")
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


        for k, v in _outer_metadata.items():
            _outer_metadata[k] = _outer_metadata[k] / grad_accum_steps

        rList = [round(value, 2) for value in earlyStopLayerDict_accum.tolist()]

        normA = _outer_metadata["_norm_attn"]
        normO = _outer_metadata["_norm_output"]
        normX = _outer_metadata["_norm_x"]
        normY = _outer_metadata["_norm_y"]
        # r_0 = _outer_metadata["routes_0"]
        # r_7 = _outer_metadata["routes_7"]

        bprint(f"@ {step} train {loss_accum.item():.4f} , allloss: {loss_accum_all.item():.4f}, dt: {dt*1000:.2f}ms, norm(attn): {normA:.4f}, norm(output): {normO:.4f}, norm(x): {normX:.4f}, norm(y): {normY:.4f}, norm:{norm:.4f}, tok/sec: {tokens_per_sec:.2f}, flops:{flops / 1e12:.2f}, batch-reuse:{timesBatchUsed}") 


if ddp:
    destroy_process_group()
exit(0)
