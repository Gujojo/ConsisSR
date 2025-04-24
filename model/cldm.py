from typing import Mapping, Any
import copy
from collections import OrderedDict

import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.modules.attention_new import HybridDecoupleTransformer

attn_type = HybridDecoupleTransformer

from ldm.modules.diffusionmodules.openaimodel_new import TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, UNetModel
from ldm.models.diffusion.ddpm_cldm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler
import random

class ControlledUnetModel(UNetModel):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     for param in self.input_blocks.parameters():
    #         param.requires_grad = False

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        decoder_channels = []
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else attn_type(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint, spatial_size=int(64*64 / (ds**2))
                            )
                        )
                        print(f'add transformer with spatial_size {int(64*64 / (ds**2))}')
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else attn_type(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint, spatial_size=int(64*64 / (ds**2))
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        print(f'add transformer with spatial_size {int(64*64 / (ds**2))}')
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        if len(decoder_channels) != 0:
            layers = []
            ch = model_channels * channel_mult[-1]
            for out_ch in decoder_channels:
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(ch, 4 * out_ch, 3, 1, 1), 
                        nn.PixelShuffle(2)
                    )
                )
                ch = out_ch

            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, 1, 1), 
                    nn.LeakyReLU(negative_slope=0.2, inplace=True), 
                    nn.Conv2d(ch, 3, 3, 1, 1)
                )
            )
            self.decoder = TimestepEmbedSequential(*layers)
            

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)
        # x = torch.cat((x, hint), dim=1)
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        if hasattr(self, 'decoder'):
            out_img = self.decoder(h, emb, context)
            return outs, out_img
        else:
            return outs, torch.zeros_like(hint)



class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_stage_config: Mapping[str, Any],
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        learning_rate: float,
        use_cfg: bool, 
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # instantiate control module
        self.control_model: ControlNet = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.sd_locked = sd_locked
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate
        self.control_scales = [1.0] * 13
        self.use_cfg = use_cfg
        
        # instantiate condition encoder, since our condition encoder has the same 
        # structure with AE encoder, we just make a copy of AE encoder. please
        # note that AE encoder's parameters has not been initialized here.
        self.cond_encoder = nn.Sequential(OrderedDict([
            ("encoder", copy.deepcopy(self.first_stage_model.encoder)), # cond_encoder.encoder
            ("quant_conv", copy.deepcopy(self.first_stage_model.quant_conv)) # cond_encoder.quant_conv
        ]))
        # for p in self.cond_encoder.parameters():
        #     p.requires_grad = True
        # frozen_module(self.first_stage_model) 
        # print('copy and frozen cond_encoder')

    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor

        return c_latent

    def get_learned_conditioning(self, c, need_hq=False, force_dropout=False):
        dropout_rate = 1.0
        assert 'txt' in c.keys()
        with torch.no_grad():
            c_txt = self.cond_stage_model.encode(c['txt'])
        if force_dropout or (self.training and random.random() > dropout_rate):
            c_txt = torch.cat([torch.zeros_like(c_txt[:, 0:1, :]), c_txt], dim=1)
        else:
            feature = self.cond_stage_model(c, need_hq)
            c_txt = torch.cat([feature, c_txt], dim=1)
        return c_txt
    
        # dropout_rate = 0.5
        # if self.training and self.use_cfg:
        #     if random.random() < dropout_rate:
        #         cond_img = c['lq'][0]
        #         c_new = {'lq': [torch.zeros_like(cond_img)], 'hq': [torch.zeros_like(cond_img)]}
        #         return self.cond_stage_model(c_new, need_hq)
        #     else:
        #         return self.cond_stage_model(c, need_hq)
        # else:
        #     return self.cond_stage_model(c, need_hq)

    # @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, _ = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        hq = batch['jpg']
        hq = einops.rearrange(hq, 'b h w c -> b c h w')
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        lq = control
        # apply condition encoder
        c_latent = self.apply_condition_encoder(control)

        return x, dict(c_crossattn=[], c_latent=[c_latent], lq=[lq], c_concat=[control], hq = [hq], txt = batch['txt'])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        # cond_controlnet = torch.cat(cond['c_emptytext'], 1)

        if cond['c_latent'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            # from thop import profile
            # macs = 0
            # params = 0
            # m, p = profile(self.control_model, (x_noisy, torch.cat(cond['c_latent'], 1), t, cond_txt))
            # macs += m
            # params += p
            control, out_img = self.control_model(
                x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
                timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]

            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps, out_img

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        raise NotImplementedError
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key)
        c_lq = c["lq"][0]
        c_latent = c["c_latent"][0]
        # c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        c_cat = c["c_concat"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        # log["control"] = c_cat
        log["decoded_control"] = (self.decode_first_stage(c_latent) + 1) / 2
        log["lq"] = (c_lq + 1) / 2
        # log["text"] = (log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16) + 1) / 2
        log["samples"] = self.sample_log(
            cond_img=c_cat,            
            steps=sample_steps
        )
        return log

    @torch.no_grad()
    def sample_log(self, cond_img, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond_img.shape
        shape = (b, self.channels, h // 8, w // 8)
        samples = sampler.sample(
            steps, shape, cond_img, positive_prompt="", negative_prompt="",
            cfg_scale=1.0, color_fix_type="adain"
        )
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        for name, param in self.control_model.named_parameters():
            # fix original cross attn for text prompt in ControlNet
            if ('attn2' in name or 'norm2' in name) and 'to_out_ip' not in name:
                pass
                # param.requires_grad = False
                # print(f'layer {name} is frozen.')
            else:
                param.requires_grad = True
                params.append(param)
                # print(f'layer {name} is optimized.')
        
        # if not self.sd_locked:
        #     params += list(self.model.diffusion_model.output_blocks.parameters())
        #     params += list(self.model.diffusion_model.out.parameters())

        for name, param in self.model.diffusion_model.named_parameters():
            # only train added params in SD
            if 'attn1.position_emb' in name or 'attnT' in name or 'normT' in name or 'to_out_ip' in name:
                param.requires_grad = True
                params.append(param)
                # print(f'layer {name} is optimized.')
            else:
                pass
                # param.requires_grad = False
                # print(f'layer {name} is frozen.')

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            # only train fine-tune layers after CLIP-img
            for name, param in self.cond_stage_model.named_parameters():
                if 'ft_layers' in name:
                    param.requires_grad = True
                    params.append(param)
                    # print(f'layer {name} is optimized.')
                else:
                    # pass
                    param.requires_grad = False
                    # print(f'layer {name} is frozen.')

        group = {'params': params, 'lr': lr}

        opt = torch.optim.AdamW([group])
        return opt

    def validation_step(self, batch, batch_idx):
        # TODO: 
        pass
