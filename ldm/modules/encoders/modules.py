import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import open_clip
from ldm.util import default, count_params
import torch.nn.functional as F
import copy
from ldm.modules.diffusionmodules.util import zero_module

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):

    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1-mask) * torch.ones_like(c)*(self.n_classes-1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class OpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    # def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, preprocess_val = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        self.device = device
        self.model = model
        # self.preprocess = preprocess_val
        self.last_process = preprocess_val.transforms[-1]

        # del model.transformer

        # self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        else:
            self.enable_train()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        
        dim = 1024
        N = 1
        self.ft_layers_4 = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, N * dim), 
            nn.GELU(), 
            Reshape(N, dim), 
            nn.Linear(dim, dim) ,
            nn.GELU(), 
            zero_module(nn.Linear(dim, dim))
        )

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def enable_train(self):
        self.model = self.model.train()
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, cond, need_hq=False):
        lq = (cond['lq'][0].to(self.device) + 1) / 2
        if lq.shape[-1] != 224 or lq.shape[-2] != 224:
            lq = F.interpolate(lq, size=(224, 224), mode='bicubic', align_corners=False)
        lq_norm = self.last_process(lq)

        if not need_hq:
            with torch.no_grad():
                image_features = self.model.encode_image(lq_norm)
            ft_features = self.ft_layers_4(image_features)
            return ft_features
        else:
            image_features = self.model.encode_image(lq_norm)
            if not hasattr(self.model, 'v_fix'):
                self.model.v_fix = copy.deepcopy(self.model.visual)
                self.model.v_fix.eval()
                for p in self.model.v_fix.parameters():
                    p.requires_grad = False
            with torch.no_grad():
                hq = (cond['hq'][0].to(self.device) + 1) / 2
                if hq.shape[-1] != 224 or lq.shape[-2] != 224:
                    hq = F.interpolate(hq, size=(224, 224), mode='bicubic', align_corners=False)
                hq_norm = self.last_process(hq)
                hq_features = self.model.v_fix(hq_norm)
                ft_features = self.ft_layers_4(image_features)
            return ft_features, image_features, hq_features


    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        tokens = open_clip.tokenize(text)
        # print(tokens)
        # z = self.encode_with_transformer(tokens.to(self.device))
        z = self.encode_with_transformer(tokens.to(next(self.model.parameters()).device))
        return z


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    # def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        version = '/my_project_path/DiffBIR-main/weights/laion2b_s32b_b79k/open_clip_pytorch_model.bin'
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        # self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        # z = self.encode_with_transformer(tokens.to(self.device))
        z = self.encode_with_transformer(tokens.to(next(self.model.parameters()).device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)

