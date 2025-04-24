import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from typing import Optional
import matplotlib.pyplot as plt
from ldm import xformers_state

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        #.........\end

        # with:
        # .........\start
        #min_encoding_indices = torch.argmin(d, dim=1)
        #z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VectorQuantizer3(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, in_c, n_e, e_dim, beta, remap=None, patch_size=2, norm_layer=nn.LayerNorm, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.in_channels = in_c
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.patch_size = patch_size
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        # patchify
        self.patch_emb = nn.Linear(patch_size*patch_size*in_c, e_dim)
        self.norm = norm_layer(e_dim)
        # unpatchify
        self.patch_unemb = nn.Linear(e_dim, patch_size*patch_size*in_c)

    def patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        assert h % p == 0
        assert w % p == 0
        patches = F.unfold(x, p, 1, 0, p)
        self.output_size = (h, w)
        patches = patches.transpose(-1, -2)
        patches = self.patch_emb(patches)
        if self.norm is not None:
            patches = self.norm(patches)
        return patches
    
    def unpatchify(self, patches):
        # b, n, d = patches.shape
        p = self.patch_size
        patches = self.patch_unemb(patches)
        patches = patches.transpose(-1, -2)
        x = F.fold(patches, self.output_size, p, 1 ,0, p)
        return x

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, num, channel) and flatten
        z = self.patchify(z)

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = self.unpatchify(z_q)
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=2, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class VectorQuantizer4(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, in_c, n_e, e_dim, beta, remap=None, kernel_size=3, norm_layer=nn.LayerNorm, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.in_channels = in_c
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        # patchify
        self.emb = nn.Conv2d(in_c, e_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = norm_layer(e_dim)
        # self.norm = L2NormalizationLayer(2)
        # unpatchify
        self.unemb = nn.Conv2d(e_dim, in_c, kernel_size, padding=(kernel_size - 1) // 2)
        self.count = np.zeros((n_e))

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        # assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, num, channel) and flatten
        z = self.emb(z)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> b (h w) c').contiguous()
        if self.norm is not None:
            z_flattened = self.norm(z_flattened)
        z_flattened = z_flattened.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        # d = -torch.einsum('bd,dn->bn', z_flattened, rearrange(F.normalize(self.embedding.weight, p=2, dim=1), 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        # for i in min_encoding_indices:
        #     self.count[i] += 1
        # if np.sum(self.count) > 4e5:
        #     positions = np.arange(len(self.count))
        #     plt.bar(positions, self.count)
        #     plt.savefig('distribution_plot.png')
        #     print('plot distribution')

        # print(set(min_encoding_indices.cpu().numpy()))
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        if return_logits:
            return  z_q.flatten(1, 2), min_encoding_indices.view((z.shape[0], -1))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = self.unemb(z_q)

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


def default(val, d):
    if val is not None:
        return val
    return d

class TransformerSALayer(nn.Module):
    def __init__(self, query_dim, context_dim=None, dim_head=None, nhead=8, dim_mlp=2048, dropout=0.0):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        dim_head =default(dim_head, query_dim // nhead)
        inner_dim = dim_head * nhead
        self.heads = nhead
        self.dim_head = dim_head
        context_dim = default(context_dim, query_dim)

        self.norm1 = nn.LayerNorm(query_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        # Implementation of Feedforward model - MLP
        self.norm2 = nn.LayerNorm(query_dim)
        self.linear1 = nn.Linear(query_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, query_dim)
        self.dropout = nn.Dropout(dropout)

        self.activation = F.gelu
        

    def forward(self, x, context=None, pos_emb: Optional[torch.Tensor] = None):
        x_ = self.norm1(x)
        context = default(context, x)
        context_ = self.norm1(context)

        if pos_emb is not None:
            q = self.to_q(x_ + pos_emb)
            k = self.to_k(context_ + pos_emb)
        else:
            q = self.to_q(x_)
            k = self.to_k(context_)
        v = self.to_v(context_)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = xformers_state.xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        x = x + self.to_out(out)

        # ffn
        x_ = self.norm2(x)
        x_ = self.linear2(self.dropout(self.activation(self.linear1(x_))))
        x = x + self.dropout(x_)
        return x


class TransformerCALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos
        
    def forward(self, tgt, tgt1, query_pos):
        
        # self attention
        q = self.norm1(tgt)
        k = v = self.norm1(tgt1)
        tgt2 = self.self_attn(q, k, v)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class VectorQuantizer5(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, in_c, n_e, e_dim, beta, kernel_size=3, n_head=8, n_layers=9, latent_size=1024, norm_layer=nn.LayerNorm, legacy=True):
        super().__init__()
        self.in_channels = in_c
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # patchify
        self.emb = nn.Conv2d(in_c, e_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = norm_layer(e_dim)
        # unpatchify
        self.unemb = nn.Conv2d(e_dim, in_c, kernel_size, padding=(kernel_size - 1) // 2)

        # codeformer
        self.position_emb = nn.Parameter(torch.zeros(latent_size, e_dim))

        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=e_dim, nhead=n_head, dim_mlp=e_dim*2, dropout=0.0) for _ in range(n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(e_dim), nn.Linear(e_dim, n_e, bias=False))

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        # assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, num, channel) and flatten
        z = self.emb(z)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> b (h w) c').contiguous()
        if self.norm is not None:
            z_flattened = self.norm(z_flattened)

        # B(HW)C
        pos_emb = self.position_emb.unsqueeze(0).repeat(z.shape[0], 1, 1)
        # z_flattened = z_flattened.permute(0,2,1)

        for layer in self.ft_layers:
            z_flattened = layer(z_flattened, query_pos=pos_emb)

        # B(HW)C
        # z_flattened = z_flattened.permute(0,2,1)

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        if return_logits:
            return  z_q.flatten(1, 2), min_encoding_indices.view((z.shape[0], -1))

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = self.unemb(z_q)
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VectorQuantizer6(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, in_c, n_e, e_dim, beta, kernel_size=3, n_head=8, n_layers=9, latent_size=1024, norm_layer=nn.LayerNorm, legacy=True):
        super().__init__()
        self.in_channels = in_c
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # patchify
        self.emb = nn.Conv2d(in_c, e_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = norm_layer(e_dim)
        # unpatchify
        self.unemb = nn.Conv2d(e_dim, in_c, kernel_size, padding=(kernel_size - 1) // 2)

        # codeformer
        self.position_emb = nn.Parameter(torch.zeros(latent_size, e_dim))

        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=e_dim, nhead=n_head, dim_mlp=e_dim*2, dropout=0.0) for _ in range(n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(e_dim), nn.Linear(e_dim, n_e, bias=False))

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, code_only=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, num, channel) and flatten
        z = self.emb(z)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> b (h w) c').contiguous()
        if self.norm is not None:
            z_flattened = self.norm(z_flattened)

        # Code Former B(HW)C
        code_emb = z_flattened + 0
        pos_emb = self.position_emb.unsqueeze(0).repeat(z.shape[0], 1, 1)

        for layer in self.ft_layers:
            code_emb = layer(code_emb, query_pos=pos_emb)

        logits = self.idx_pred_layer(code_emb)

        # (BHW)C
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        z_q = self.get_codebook_entry(top_idx, shape=z.shape)
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = self.unemb(z_q)

        # # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # z_q = self.unemb(z_q)
        if code_only:
            return z_q, logits, z_flattened
        else:
            return z_q


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VectorQuantizer7(nn.Module):
    def __init__(self, in_c, n_e, e_dim, beta, kernel_size=3, n_head=8, n_layers=5, latent_size=1024, norm_layer=nn.LayerNorm, legacy=True):
        super().__init__()
        self.in_channels = in_c
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # patchify
        self.emb = nn.Conv2d(in_c, e_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = norm_layer(e_dim)
        # self.norm = L2NormalizationLayer(2)
        # unpatchify
        self.unemb = nn.Conv2d(e_dim, in_c, kernel_size, padding=(kernel_size - 1) // 2)

        # codeformer
        self.position_emb = nn.Parameter(torch.zeros(latent_size, e_dim))

        layer_list_sa = []
        layer_list_ca = []
        for _ in range(n_layers):
            layer_list_sa.append(TransformerSALayer(query_dim=e_dim, nhead=n_head, dim_mlp=e_dim*2, dropout=0.0))
            layer_list_ca.append(TransformerSALayer(query_dim=e_dim, nhead=n_head, dim_mlp=e_dim*2, dropout=0.0))

        self.ft_layers_sa = nn.ModuleList(layer_list_sa)
        self.ft_layers_ca = nn.ModuleList(layer_list_ca)

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(e_dim), nn.Linear(e_dim, n_e, bias=False))

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, code_only=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, num, channel) and flatten
        z = self.emb(z)
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> b (h w) c').contiguous()
        if self.norm is not None:
            z_flattened = self.norm(z_flattened)

        if z_flattened.shape[1] != self.position_emb.shape[0]:
            self.position_emb.data = F.interpolate(self.position_emb.data.unsqueeze(0).unsqueeze(0), z_flattened.shape[1:], mode='bilinear').squeeze()
            print(f'Infer: Position emb shape is changed to {self.position_emb.shape}')

        # Code Former B(HW)C
        code_emb = z_flattened + 0
        pos_emb = self.position_emb.unsqueeze(0).repeat(z.shape[0], 1, 1)
        batch_emb= self.embedding.weight.unsqueeze(0).repeat(z.shape[0], 1, 1)
        for i in range(len(self.ft_layers_sa)):
            code_emb = self.ft_layers_sa[i](code_emb, pos_emb=pos_emb)
            code_emb = self.ft_layers_ca[i](code_emb, context=batch_emb)

        logits = self.idx_pred_layer(code_emb)

        # (BHW)C
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        z_q = self.get_codebook_entry(top_idx, shape=z.shape)
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        z_q = self.unemb(z_q)

        # # reshape back to match original input shape
        # z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # z_q = self.unemb(z_q)
        if code_only:
            return z_q, logits, z_flattened
        else:
            return z_q


    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    # def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        
    #     if 'position_emb' in state_dict.keys():
    #         data_ori = state_dict['position_emb']
    #         super().load_state_dict(state_dict, strict=strict, assign=assign)

    #         self.position_emb.data = F.interpolate(data_ori.unsqueeze(0).unsqueeze(0), [4096, self.e_dim], mode='bilinear').squeeze()
    #         print(f'Position emb shape is changed to {self.position_emb.shape}')
    #     else:
    #         super().load_state_dict(state_dict, strict=strict, assign=assign)
        