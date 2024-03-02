# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self

from lit_gpt.config import Config
import copy
from copy import deepcopy

def copy_param(model1, model2):
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        param1.data.copy_(param2.data)

# class IntentionGPT(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         assert config.padded_vocab_size is not None
#         self.config = config

#         self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)

#         encoder_layer_num = 1
#         self.encoder_layer_num = encoder_layer_num
#         n_action_embd = config.n_embd
#         self.sentence_action = True
#         self.finetune = False
#         self.state_encoder = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(encoder_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.action_encoder = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(encoder_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.mean_layer = nn.Linear(config.n_embd, n_action_embd)
#         self.logvar_layer = nn.Linear(config.n_embd, n_action_embd)
#         concat_block_config = copy.deepcopy(config)
#         concat_block_config.input_n_embd = concat_block_config.n_embd + n_action_embd
#         self.concat_block = Block(concat_block_config)
#         decoder_layer_num = config.n_layer - encoder_layer_num
#         self.decoder = nn.ModuleDict(
#             dict(
#                 h=nn.ModuleList(Block(config) for _ in range(decoder_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
#         self.max_seq_length = self.config.block_size
#         self.mask_cache: Optional[torch.Tensor] = None
        
#         if self.finetune:
#             self.decoder.eval()
#             self.state_encoder.eval()

#     @property
#     def max_seq_length(self) -> int:
#         return self._max_seq_length

#     @max_seq_length.setter
#     def max_seq_length(self, value: int) -> None:
#         """
#         When doing inference, the sequences used might be shorter than the model's context length.
#         This allows setting a smaller number to avoid allocating unused memory
#         """
#         if value > self.config.block_size:
#             raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
#         self._max_seq_length = value
#         if not hasattr(self, "cos"):
#             # first call
#             cos, sin = self.rope_cache()
#             self.register_buffer("cos", cos, persistent=False)
#             self.register_buffer("sin", sin, persistent=False)
#         # override
#         elif value != self.cos.size(0):
#             self.cos, self.sin = self.rope_cache(device=self.cos.device)
#         # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
#         # if the kv cache is expected

#     def reset_parameters(self) -> None:
#         # Trigger resetting the rope-cache
#         self.cos, self.sin = self.rope_cache()

#     def _init_weights(self, module: nn.Module) -> None:
#         """Meant to be used with `gpt.apply(gpt._init_weights)`."""
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

#     def reparameterization(self, mean, var):
#         epsilon = torch.randn_like(var)
#         z = mean + var*epsilon
#         return z
    
#     def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, train_mode=False) -> torch.Tensor:
#         T = idx.size(1)
#         if self.max_seq_length < T:
#             raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

#         if input_pos is not None:  # use the kv cache
#             cos = self.cos.index_select(0, input_pos)
#             sin = self.sin.index_select(0, input_pos)
#             if self.mask_cache is None:
#                 raise TypeError("You need to call `gpt.set_kv_cache()`")
#             mask = self.mask_cache.index_select(2, input_pos)
#         else:
#             cos = self.cos[:T]
#             sin = self.sin[:T]
#             mask = None

#         x = self.state_encoder.wte(idx)  # token embeddings of shape (b, t, n_embd)
#         action_x = self.action_encoder.wte(idx)
#         for block_state, block_action in zip(self.state_encoder.h, self.action_encoder.h):
#             x = block_state(x, cos, sin, mask, input_pos)
#             aciton_x = block_action(action_x, cos, sin, mask, input_pos)
            
#         if self.sentence_action:
#             action_x = action_x[:, -1:, :].repeat(1, T, 1)
#         else:
#             action_x[:, :-1] = action_x[:, 1:]

#         mean, logvar = self.mean_layer(aciton_x), self.logvar_layer(aciton_x)
#         z = self.reparameterization(mean, logvar)

#         x = torch.cat([x, z], dim=-1)
#         for block in enumerate(self.decoder.h):
#             x = block(x, cos, sin, mask, input_pos)
#         x = self.decoder.ln_f(x)
#         # TODO need to return action
#         if not train_mode:
#             return self.lm_head(x)
        
#         return self.lm_head(x), {"mean": mean, "logvar": logvar, "z": z}  # (b, t, vocab_size)
    
#     def load_from_gpt(self, gpt_model):
#         self.state_encoder.wte.load_state_dict(gpt_model.transformer.wte.state_dict())
#         for idx, block in enumerate(self.state_encoder.h):
#             block.load_state_dict(gpt_model.transformer.h[idx].state_dict())
            
#         for idx, block in enumerate(self.decoder.h):
#             block.load_state_dict(gpt_model.transformer.h[idx + self.encoder_layer_num].state_dict())
#         self.decoder.ln_f.load_state_dict(gpt_model.transformer.ln_f.state_dict())
#         self.lm_head.load_state_dict(gpt_model.lm_head.state_dict()) 

#     @classmethod
#     def from_name(cls, name: str, **kwargs: Any) -> Self:
#         return cls(Config.from_name(name, **kwargs))

#     def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         return build_rope_cache(
#             seq_len=self.max_seq_length,
#             n_elem=self.config.rope_n_elem,
#             device=device,
#             condense_ratio=self.config.rope_condense_ratio,
#             base=self.config.rope_base,
#         )

#     def set_kv_cache(
#         self,
#         batch_size: int,
#         rope_cache_length: Optional[int] = None,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         if rope_cache_length is None:
#             rope_cache_length = self.cos.size(-1)
#         max_seq_length = self.max_seq_length

#         # initialize the kv cache for all blocks
#         for block in self.transformer.h:
#             block.attn.kv_cache = block.attn.build_kv_cache(
#                 batch_size, max_seq_length, rope_cache_length, device, dtype
#             )

#         if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
#             # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
#             # for the kv-cache support (only during inference), we only create it in that situation
#             self.mask_cache = build_mask_cache(max_seq_length, device)

#     def clear_kv_cache(self) -> None:
#         self.mask_cache = None
#         for block in self.transformer.h:
#             block.attn.kv_cache = None

import einops
from einops.layers.torch import Rearrange
from einops import rearrange

# class CrossAttention(nn.Module):
#     def __init__(self, heads=4, embd_dim=64, dropout=0.):
#         super().__init__()
#         self.heads = heads

#         self.to_q = nn.Linear(embd_dim, embd_dim, bias=False)
#         self.to_k = nn.Linear(embd_dim, embd_dim, bias=False)
#         self.to_v = nn.Linear(embd_dim, embd_dim, bias=False)

#         self.attention = nn.MultiheadAttention(embd_dim, batch_first=True, num_heads=heads)

#     # def forward(self, x, context=None, mask=None):
#     #     assert x.shape[1] == context.shape[1]
#     #     q = self.to_q(x)
#     #     k = self.to_k(torch.cat([x, context], dim=1))
#     #     v = self.to_v(torch.cat([x, context], dim=1))
        
#     #     attn_mask_x = (torch.triu(torch.ones(q.shape[1], q.shape[1])) == 1).transpose(0, 1).to(q.device)
#     #     attn_mask_c = (torch.eye(q.shape[1]) == 1).to(q.device)
#     #     attn_mask = torch.cat([attn_mask_x, attn_mask_c], dim=-1)
        
#     #     attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
#     #     out, _ = self.attention(q, k, v, attn_mask=attn_mask)
        
#     #     return out
    
#     def forward(self, x, context=None, mask=None):
#         assert x.shape[1] == context.shape[1]
#         q = self.to_q(x)
#         k = self.to_k(context)
#         v = self.to_v(context)
        
#         attn_mask = (torch.triu(torch.ones(q.shape[1], q.shape[1])) == 1).transpose(0, 1).to(q.device)
        
#         attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
#         out, _ = self.attention(q, k, v, attn_mask=attn_mask)
        
#         return out


# class IntentionGPT(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         assert config.padded_vocab_size is not None
#         self.config = config
        
#         self.mask_ratio = 0.2

#         self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        
#         self.enc_layer_num = 1
#         self.dyna_layer_num = 1
#         self.dec_layer_num = config.n_layer - self.enc_layer_num - self.dyna_layer_num
#         self.hidden_num = 64
        
#         # self.dec_layer_num = 1
#         self.transformer_enc = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(self.enc_layer_num)),
#             )
#         )
#         self.transformer_act = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(self.enc_layer_num)),
#             )
#         )
#         self.transformer_bc = nn.ModuleDict(
#             dict(
#                 wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
#                 h=nn.ModuleList(Block(config) for _ in range(self.enc_layer_num)),
#             )
#         )
#         self.mean_layer = nn.Linear(config.n_embd, self.hidden_num)
#         self.logvar_layer = nn.Linear(config.n_embd, self.hidden_num)
        
#         self.trans_layer = nn.Linear(self.hidden_num, config.n_embd)
#         # self.transformer_dyna = CrossAttention(heads=4 if config.n_embd % 4 == 0 else 1, embd_dim=config.n_embd)
        
#         self.transformer_dyna = nn.ModuleDict(
#             dict(
#                 h=nn.ModuleList(Block(config, attn_type='cross') for _ in range(self.dyna_layer_num)),
#             )
#         )
        
#         self.transformer_dec = nn.ModuleDict(
#             dict(
#                 h=nn.ModuleList(Block(config) for _ in range(self.dec_layer_num)),
#                 ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
#             )
#         )
        
#         self.max_seq_length = self.config.block_size
#         self.mask_cache: Optional[torch.Tensor] = None

#     @property
#     def max_seq_length(self) -> int:
#         return self._max_seq_length

#     @max_seq_length.setter
#     def max_seq_length(self, value: int) -> None:
#         """
#         When doing inference, the sequences used might be shorter than the model's context length.
#         This allows setting a smaller number to avoid allocating unused memory
#         """
#         if value > self.config.block_size:
#             raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
#         self._max_seq_length = value
#         if not hasattr(self, "cos"):
#             # first call
#             cos, sin = self.rope_cache()
#             self.register_buffer("cos", cos, persistent=False)
#             self.register_buffer("sin", sin, persistent=False)
#         # override
#         elif value != self.cos.size(0):
#             self.cos, self.sin = self.rope_cache(device=self.cos.device)
#         # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
#         # if the kv cache is expected

#     def reset_parameters(self) -> None:
#         # Trigger resetting the rope-cache
#         self.cos, self.sin = self.rope_cache()

#     def _init_weights(self, module: nn.Module) -> None:
#         """Meant to be used with `gpt.apply(gpt._init_weights)`."""
#         if isinstance(module, nn.Linear):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#             if module.bias is not None:
#                 torch.nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.Embedding):
#             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
#     def reparameterization(self, mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         epsilon = torch.randn_like(std)
#         z = mean + std*epsilon
#         return z

#     def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, train_mode=False) -> torch.Tensor:
#         T = idx.size(1)
#         if self.max_seq_length < T:
#             raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

#         if input_pos is not None:  # use the kv cache
#             cos = self.cos.index_select(0, input_pos)
#             sin = self.sin.index_select(0, input_pos)
#             if self.mask_cache is None:
#                 raise TypeError("You need to call `gpt.set_kv_cache()`")
#             mask = self.mask_cache.index_select(2, input_pos)
#         else:
#             cos = self.cos[:T]
#             sin = self.sin[:T]
#             mask = None

#         x = self.transformer_enc.wte(idx)  # token embeddings of shape (b, t, n_embd)
#         x_act = self.transformer_act.wte(idx)  # token embeddings of shape (b, t, n_embd)
#         x_bc = self.transformer_bc.wte(idx)  # token embeddings of shape (b, t, n_embd)
#         for block, block_a, block_b in zip(self.transformer_enc.h, self.transformer_act.h, self.transformer_bc.h):
#             x = block(x, cos, sin, mask, input_pos)
#             x_act = block_a(x_act, cos, sin, mask, input_pos)
#             x_bc = block_b(x_bc, cos, sin, mask, input_pos)
#         x_act[:, :-1] = x_act[:, 1:]
            
#         # version 1
#         # x_act = x_act[:, -1:, :]
#         # mean, logvar = self.mean_layer(x_act), self.logvar_layer(x_act)
#         # z = self.reparameterization(mean, torch.exp(logvar))
#         # x = torch.cat([z, x], dim=1)
#         # cos_ = torch.cat([torch.zeros_like(cos[:1]), cos], dim=0)
#         # sin_ = torch.cat([torch.zeros_like(sin[:1]), sin], dim=0)
#         # mask_ = torch.cat([torch.ones_like(mask[:1]), mask], dim=0) if mask is not None else None
#         # input_pos_ = torch.cat([torch.ones_like(input_pos[:1]), input_pos], dim=0) if input_pos is not None else None
#         # for block in self.transformer_dyna.h:
#         #     x = block(x, cos_, sin_, mask_, input_pos_)
#         # x = x[:, 1:]
        
#         # version 2
#         # mean, logvar = self.mean_layer(x_act), self.logvar_layer(x_act)
#         # z = self.reparameterization(mean, torch.exp(logvar))
#         # x = torch.cat([x, z], dim=-1)
#         # x = self.concat_layer(x)
#         # for block in self.transformer_dec.h:
#         #     x = block(x, cos, sin, mask, input_pos)
        
#         # version 3
#         mean, logvar = self.mean_layer(x_act), self.logvar_layer(x_act)
#         mean_bc, logvar_bc = self.mean_layer(x_bc), self.logvar_layer(x_bc)
#         action = self.reparameterization(mean, logvar)
        
#         action = self.trans_layer(action)
#         # random_hidden_mask = torch.rand(z.shape[0], z.shape[1])
#         # z[random_hidden_mask < self.mask_ratio] = 0
        
#         for block in self.transformer_dyna.h:
#             x = block([action, x], cos, sin, mask, input_pos)
#         # x = self.cross_attention_layer(action, x)
        
#         for block in self.transformer_dec.h:
#             x = block(x, cos, sin, mask, input_pos)
            
#         x = self.transformer_dec.ln_f(x)
#         if not train_mode:
#             return self.lm_head(x)
        
#         ent = 0.5 * torch.log(2 * torch.pi * torch.e * torch.exp(logvar))
        
#         return self.lm_head(x), {"mean": mean, 
#                                  "logvar": logvar, 
#                                  "mean_bc": mean_bc, 
#                                  "logvar_bc": logvar_bc, 
#                                  "z": action, 
#                                  "entropy_mean": ent.mean(), 
#                                  "entropy_std": ent.std(), 
#                                  "entropy_max": ent.max(dim=-1)[0].mean(), 
#                                  "entropy_min": ent.min(dim=-1)[0].mean(),
#                                  "mean_mean": mean.mean(), 
#                                  "mean_std": mean.std(), 
#                                  "mean_max": mean.max(dim=-1)[0].mean(), 
#                                  "mean_min": mean.min(dim=-1)[0].mean(),
#                                  "std_mean": torch.exp(0.5 * logvar).mean(), 
#                                  "std_std": torch.exp(0.5 * logvar).std(), 
#                                  "std_max": torch.exp(0.5 * logvar).max(dim=-1)[0].mean(), 
#                                  "std_min": torch.exp(0.5 * logvar).min(dim=-1)[0].mean(),}  # (b, t, vocab_size)

#     @classmethod
#     def from_name(cls, name: str, **kwargs: Any) -> Self:
#         return cls(Config.from_name(name, **kwargs))

#     def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         return build_rope_cache(
#             seq_len=self.max_seq_length,
#             n_elem=self.config.rope_n_elem,
#             device=device,
#             condense_ratio=self.config.rope_condense_ratio,
#             base=self.config.rope_base,
#         )

#     def set_kv_cache(
#         self,
#         batch_size: int,
#         rope_cache_length: Optional[int] = None,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> None:
#         if rope_cache_length is None:
#             rope_cache_length = self.cos.size(-1)
#         max_seq_length = self.max_seq_length

#         # initialize the kv cache for all blocks
#         for block in self.transformer.h:
#             block.attn.kv_cache = block.attn.build_kv_cache(
#                 batch_size, max_seq_length, rope_cache_length, device, dtype
#             )

#         if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
#             # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
#             # for the kv-cache support (only during inference), we only create it in that situation
#             self.mask_cache = build_mask_cache(max_seq_length, device)

#     def clear_kv_cache(self) -> None:
#         self.mask_cache = None
#         for block in self.transformer.h:
#             block.attn.kv_cache = None


class IntentionGPT(nn.Module):
    def __init__(self, config: Config, hidden_dim: int = 64) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        
        self.mask_ratio = 0.2

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        
        self.enc_layer_num = 1
        self.dyna_layer_num = 1
        self.dec_layer_num = config.n_layer - self.enc_layer_num - self.dyna_layer_num
        self.hidden_num = hidden_dim
        
        # self.dec_layer_num = 1
        self.transformer_enc = nn.ModuleDict(
            dict(
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            )
        )
        self.transformer_bc = nn.ModuleDict(
            dict(
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            )
        )
        self.mean_layer = nn.Linear(config.n_embd, self.hidden_num, bias=config.lm_head_bias)
        self.logvar_layer = nn.Linear(config.n_embd, self.hidden_num, bias=config.lm_head_bias)
        self.trans_layer = nn.Linear(self.hidden_num, config.n_embd, bias=config.lm_head_bias)
        
        self.transformer_dec = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None
        
        # self.bc_transformer_dec = nn.ModuleDict(
        #     dict(
        #         wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
        #         h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
        #         ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
        #     )
        # )
        # self.bc_mean_layer = nn.Linear(config.n_embd, self.hidden_num, bias=config.lm_head_bias)
        # self.bc_logvar_layer = nn.Linear(config.n_embd, self.hidden_num, bias=config.lm_head_bias)
        # self.bc_trans_layer = nn.Linear(self.hidden_num, config.n_embd, bias=config.lm_head_bias)
        # self.bc_lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        
        # self.bc_transformer_dec.requires_grad_(False)
        # self.bc_mean_layer.requires_grad_(False)
        # self.bc_logvar_layer.requires_grad_(False)
        # self.bc_trans_layer.requires_grad_(False)
        # self.bc_lm_head.requires_grad_(False)
        
    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std*epsilon
        return z
    
    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, train_mode=False, action_copy=False, action_bias=None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
            
        if action_bias is not None:
            # self.transformer_dec.requires_grad_(False)
            # self.transformer_bc.requires_grad_(False)
            # self.transformer_enc.requires_grad_(False)
            
            # self.lm_head.requires_grad_(False)
            # self.trans_layer.requires_grad_(False)
            # self.mean_layer.requires_grad_(False)
            # self.logvar_layer.requires_grad_(False)
            
            x_a = self.transformer_dec.wte(idx) 
            for block_a in self.transformer_enc.h:
                x_a = block_a(x_a, cos, sin, mask, input_pos)
                
            x_a[:, :-1] = x_a[:, 1:].clone()
            mean, logvar = self.mean_layer(x_a), self.logvar_layer(x_a)
            action = self.reparameterization(mean, logvar)
            if type(action_bias) != float:
                action = action_bias
            elif type(action_bias) == bool:
                return action
            else:
                action += action_bias
            
            x_a = self.trans_layer(action)
            x_s = self.transformer_dec.wte(idx)
            bs, lens, dims = x_a.size()
            x = torch.stack([x_s, x_a], dim=1).permute(0, 2, 1, 3).reshape(bs, 2*lens, -1)
            cos = torch.stack([cos, cos], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
            sin = torch.stack([sin, sin], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
            if input_pos is not None:
                input_pos = torch.stack([input_pos, input_pos], dim=0).permute(1, 0).reshape(2*lens)
                mask = torch.stack([mask, mask], dim=2).permute(0, 1, 3, 2, 4).reshape(bs, 1, 2*lens, -1)

            for block in self.transformer_dec.h:
                x = block(x, cos, sin, mask, input_pos)    
            x = x[:, 1::2]
                
            x = self.transformer_dec.ln_f(x)
            x = self.lm_head(x)
            
        elif not action_copy:
            # self.transformer_dec.requires_grad_(True)
            # self.transformer_bc.requires_grad_(False)
            # self.transformer_enc.requires_grad_(True)
            
            # self.lm_head.requires_grad_(True)
            # self.trans_layer.requires_grad_(True)
            # self.mean_layer.requires_grad_(True)
            # self.logvar_layer.requires_grad_(True)
            
            x_a = self.transformer_dec.wte(idx) 
            for block_a in self.transformer_enc.h:
                x_a = block_a(x_a, cos, sin, mask, input_pos)
            x_a[:, :-1] = x_a[:, 1:]
            
            mean, logvar = self.mean_layer(x_a), self.logvar_layer(x_a)
            action = self.reparameterization(mean, logvar)
            if action_bias is not None and type(action_bias) != float:
                action = action_bias
            
            x_a = self.trans_layer(action)
            x_s = self.transformer_dec.wte(idx)
            bs, lens, dims = x_a.size()
            x = torch.stack([x_s, x_a], dim=1).permute(0, 2, 1, 3).reshape(bs, 2*lens, -1)
            cos = torch.stack([cos, cos], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
            sin = torch.stack([sin, sin], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
            if input_pos is not None:
                input_pos = torch.stack([input_pos, input_pos], dim=0).permute(1, 0).reshape(2*lens)
                mask = torch.stack([mask, mask], dim=2).permute(0, 1, 3, 2, 4).reshape(bs, 1, 2*lens, -1)

            for block in self.transformer_dec.h:
                x = block(x, cos, sin, mask, input_pos)    
            x = x[:, 1::2]
                
            x = self.transformer_dec.ln_f(x)
            x = self.lm_head(x)
        else:
            # copy_param(self.bc_transformer_dec, self.transformer_dec)
            # copy_param(self.bc_lm_head, self.lm_head)
            # copy_param(self.bc_trans_layer, self.trans_layer)
            # copy_param(self.bc_mean_layer, self.mean_layer)
            # copy_param(self.bc_logvar_layer, self.logvar_layer)
            # self.bc_transformer_dec.load_state_dict(self.transformer_dec.state_dict())
            # self.bc_mean_layer.load_state_dict(self.mean_layer.state_dict())
            # self.bc_logvar_layer.load_state_dict(self.logvar_layer.state_dict())
            # self.bc_trans_layer.load_state_dict(self.trans_layer.state_dict())
            # self.bc_lm_head.load_state_dict(self.lm_head.state_dict())
            
            # self.transformer_dec.requires_grad_(False)
            # self.transformer_bc.requires_grad_(True)
            # self.transformer_enc.requires_grad_(False)
            # self.lm_head.requires_grad_(False)
            # self.trans_layer.requires_grad_(False)
            # self.mean_layer.requires_grad_(False)
            # self.logvar_layer.requires_grad_(False)
            
            x_a = self.transformer_dec.wte(idx.detach()) 
            for block_a in self.transformer_bc.h:
                x_a = block_a(x_a, cos, sin, mask, input_pos)
            
            mean, logvar = self.mean_layer(x_a), self.logvar_layer(x_a)
            action = self.reparameterization(mean, logvar)
            
            x_a = self.trans_layer(action)
            x_s = self.transformer_dec.wte(idx.detach())
            bs, lens, dims = x_a.size()
            x = torch.stack([x_s, x_a], dim=1).permute(0, 2, 1, 3).reshape(bs, 2*lens, -1)
            cos = torch.stack([cos, cos], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
            sin = torch.stack([sin, sin], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
            if input_pos is not None:
                input_pos = torch.stack([input_pos, input_pos], dim=0).permute(1, 0).reshape(2*lens)
                mask = torch.stack([mask, mask], dim=2).permute(0, 1, 3, 2, 4).reshape(bs, 1, 2*lens, -1)

            for block in self.transformer_dec.h:
                x = block(x, cos, sin, mask, input_pos)    
            x = x[:, 1::2]
                
            x = self.transformer_dec.ln_f(x)
            x = self.lm_head(x)  
            
        if not train_mode:
            return x
        
        ent = 0.5 * torch.log(2 * torch.pi * torch.e * torch.exp(logvar))
        return x, {"mean": mean, 
                                 "logvar": logvar, 
                                 "z": action, 
                                 "entropy_mean": ent.mean(), 
                                 "entropy_std": ent.std(), 
                                 "entropy_max": ent.max(dim=-1)[0].mean(), 
                                 "entropy_min": ent.min(dim=-1)[0].mean(),
                                 "mean_mean": mean.mean(), 
                                 "mean_std": mean.std(), 
                                 "mean_max": mean.max(dim=-1)[0].mean(), 
                                 "mean_min": mean.min(dim=-1)[0].mean(),
                                 "std_mean": torch.exp(0.5 * logvar).mean(), 
                                 "std_std": torch.exp(0.5 * logvar).std(), 
                                 "std_max": torch.exp(0.5 * logvar).max(dim=-1)[0].mean(), 
                                 "std_min": torch.exp(0.5 * logvar).min(dim=-1)[0].mean(),}  # (b, t, vocab_size)


    # def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None, train_mode=False, action_bias=None) -> torch.Tensor:
    #     T = idx.size(1)
    #     if self.max_seq_length < T:
    #         raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

    #     if input_pos is not None:  # use the kv cache
    #         cos = self.cos.index_select(0, input_pos)
    #         sin = self.sin.index_select(0, input_pos)
    #         if self.mask_cache is None:
    #             raise TypeError("You need to call `gpt.set_kv_cache()`")
    #         mask = self.mask_cache.index_select(2, input_pos)
    #     else:
    #         cos = self.cos[:T]
    #         sin = self.sin[:T]
    #         mask = None

    #     x_a = self.transformer_dec.wte(idx)  # token embeddings of shape (b, t, n_embd)
    #     # x_b = self.transformer_bc.wte(idx)  # token embeddings of shape (b, t, n_embd)
    #     # for block_a, block_b in zip(self.transformer_enc.h, self.transformer_bc.h):
    #     #     x_a = block_a(x_a, cos, sin, mask, input_pos)
    #         # x_b = block_b(x_b, cos, sin, mask, input_pos)
    #     for block_a in self.transformer_enc.h:
    #         x_a = block_a(x_a, cos, sin, mask, input_pos)
            
    #     if action_bias is not None:
    #         x_a[:, :-1] = x_a[:, 1:].clone()
    #     else:
    #         x_a[:, :-1] = x_a[:, 1:]
        
    #     mean, logvar = self.mean_layer(x_a), self.logvar_layer(x_a)
    #     # mean_bc, logvar_bc = self.mean_layer(x_b), self.logvar_layer(x_b)
    #     action = self.reparameterization(mean, logvar)
    #     # action = action + action_bias if action_bias is not None else action
    #     # assert action.shape == action_bias.shape
    #     if action_bias is not None and type(action_bias) != float:
    #         action = action_bias
        
    #     x_a = self.trans_layer(action)
    #     x_s = self.transformer_dec.wte(idx)
    #     bs, lens, dims = x_a.size()
    #     x = torch.stack([x_s, x_a], dim=1).permute(0, 2, 1, 3).reshape(bs, 2*lens, -1)
    #     cos = torch.stack([cos, cos], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
    #     sin = torch.stack([sin, sin], dim=0).permute(1, 0, 2).reshape(2*lens, -1)
    #     if input_pos is not None:
    #         input_pos = torch.stack([input_pos, input_pos], dim=0).permute(1, 0).reshape(2*lens)
    #         mask = torch.stack([mask, mask], dim=2).permute(0, 1, 3, 2, 4).reshape(bs, 1, 2*lens, -1)

    #     for block in self.transformer_dec.h:
    #         x = block(x, cos, sin, mask, input_pos)    
    #     x = x[:, 1::2]
            
    #     x = self.transformer_dec.ln_f(x)
    #     if not train_mode:
    #         return self.lm_head(x)
        
    #     ent = 0.5 * torch.log(2 * torch.pi * torch.e * torch.exp(logvar))
        
    #     return self.lm_head(x), {"mean": mean, 
    #                              "logvar": logvar, 
    #                             #  "mean_bc": mean_bc, 
    #                             #  "logvar_bc": logvar_bc, 
    #                              "z": action, 
    #                              "entropy_mean": ent.mean(), 
    #                              "entropy_std": ent.std(), 
    #                              "entropy_max": ent.max(dim=-1)[0].mean(), 
    #                              "entropy_min": ent.min(dim=-1)[0].mean(),
    #                              "mean_mean": mean.mean(), 
    #                              "mean_std": mean.std(), 
    #                              "mean_max": mean.max(dim=-1)[0].mean(), 
    #                              "mean_min": mean.min(dim=-1)[0].mean(),
    #                              "std_mean": torch.exp(0.5 * logvar).mean(), 
    #                              "std_std": torch.exp(0.5 * logvar).std(), 
    #                              "std_max": torch.exp(0.5 * logvar).max(dim=-1)[0].mean(), 
    #                              "std_min": torch.exp(0.5 * logvar).min(dim=-1)[0].mean(),}  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block_e, block_d in zip(self.transformer_enc.h, self.transformer_dec.h):
            block_e.attn.kv_cache = block_e.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )
            block_d.attn.kv_cache = block_d.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None



class GPT(nn.Module):
    def __init__(self, config: Config, num_layer=None) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        if num_layer is not None:
            config.n_layer = num_layer

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache()

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")

        if input_pos is not None:  # use the kv cache
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
        )

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        for block in self.transformer.h:
            block.attn.kv_cache = None



class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_1 = self.norm_1(x)
        h = self.attn(n_1, cos, sin, mask, input_pos)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = self.mlp(n_2) + h + x
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            x = h + x
            x = self.mlp(self.norm_2(x)) + x
        return x
# class Block(nn.Module):
#     def __init__(self, config: Config, attn_type='self') -> None:
#         super().__init__()
#         self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
#         self.attn = CausalSelfAttention(config) if attn_type == 'self' else CausalCrossAttention(config)
#         self.norm_2 = None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps)
#         self.mlp = config.mlp_class(config)

#         self.config = config
#         self.attn_type = attn_type

#     def forward(
#         self,
#         x: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         input_pos: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         if self.attn_type == 'self':
#             n_1 = self.norm_1(x)
#         else:
#             n_1 = [self.norm_1(x[0]), self.norm_2(x[1])]
#             x = x[1]
#         h = self.attn(n_1, cos, sin, mask, input_pos)
#         if self.config.parallel_residual:
#             n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
#             x = self.mlp(n_2) + h + x
#         else:
#             if self.config.shared_attention_norm:
#                 raise NotImplementedError(
#                     "No checkpoint amongst the ones we support uses this configuration"
#                     " (non-parallel residual and shared attention norm)."
#                 )
#             x = h + x
#             x = self.mlp(self.norm_2(x)) + x
#         return x

# class CausalCrossAttention(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         shape_q = (config.n_head) * config.head_size
#         shape_v = (2 * config.n_query_groups) * config.head_size
#         # key, query, value projections for all heads, but in a batch
#         self.attn_q = nn.Linear(config.n_embd, shape_q, bias=config.bias)
#         self.attn_kv = nn.Linear(config.n_embd, shape_v, bias=config.bias)
#         # output projection
#         self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
#         # disabled by default
#         self.kv_cache: Optional[KVCache] = None

#         self.config = config

#     def forward(
#         self,
#         x: torch.Tensor,
#         cos: torch.Tensor,
#         sin: torch.Tensor,
#         mask: Optional[torch.Tensor] = None,
#         input_pos: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         xq = x[0]
#         xkv = x[1]
        
#         B, T, C = xq.size()  # batch size, sequence length, embedding dimensionality (n_embd)

#         q = self.attn_q(xq)
#         q_per_kv = self.config.n_head // self.config.n_query_groups
#         kv = self.attn_kv(xkv)
#         q = q.view(B, T, self.config.n_query_groups, q_per_kv, self.config.head_size)
#         q = q.permute(0, 2, 3, 1, 4)
#         kv = kv.view(B, T, self.config.n_query_groups, 2, self.config.head_size)
#         kv = kv.permute(0, 2, 3, 1, 4)
#         k, v = kv.split((1, 1), dim=2)

#         # maybe repeat k and v if for the non multi-head attention cases
#         # training: flash attention requires it
#         # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
#         if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
#             k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
#             v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

#         q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
#         k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
#         v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

#         q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
#         k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
#         q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
#         k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

#         if input_pos is not None:
#             if not isinstance(self.kv_cache, KVCache):
#                 raise TypeError("You need to call `gpt.set_kv_cache()`")
#             k, v = self.kv_cache(input_pos, k, v)

#         y = self.scaled_dot_product_attention(q, k, v, mask)

#         y = y.reshape(B, T, self.config.n_embd)  # re-assemble all head outputs side by side

#         # output projection
#         return self.proj(y)

#     def scaled_dot_product_attention(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         scale = 1.0 / math.sqrt(self.config.head_size)
#         y = torch.nn.functional.scaled_dot_product_attention(
#             q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
#         )
#         return y.transpose(1, 2)

#     def build_kv_cache(
#         self,
#         batch_size: int,
#         max_seq_length: int,
#         rope_cache_length: Optional[int] = None,
#         device: Optional[torch.device] = None,
#         dtype: Optional[torch.dtype] = None,
#     ) -> "KVCache":
#         heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
#         v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
#         if rope_cache_length is None:
#             if self.config.rotary_percentage != 1.0:
#                 raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
#             k_shape = v_shape
#         else:
#             k_shape = (
#                 batch_size,
#                 heads,
#                 max_seq_length,
#                 rope_cache_length + self.config.head_size - self.config.rope_n_elem,
#             )
#         return KVCache(k_shape, v_shape, device=device, dtype=dtype)



class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        # maybe repeat k and v if for the non multi-head attention cases
        # training: flash attention requires it
        # inference: multi-query would require a full kv cache so avoid it to limit its memory usage
        if self.config.n_query_groups != self.config.n_head and (input_pos is None or self.config.n_query_groups != 1):
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            k, v = self.kv_cache(input_pos, k, v)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.n_embd)  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        return y.transpose(1, 2)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_head
        v_shape = (batch_size, heads, max_seq_length, self.config.head_size)
        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError("Please pass the `rope_cache_length=gpt.cos.size(-1)` value")
            k_shape = v_shape
        else:
            k_shape = (
                batch_size,
                heads,
                max_seq_length,
                rope_cache_length + self.config.head_size - self.config.rope_n_elem,
            )
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x, approximate=self.config.gelu_approximate)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(LLaMAMLP(config) for _ in range(config.n_expert))

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(router, self.config.n_expert_per_token)  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(self.config.n_expert, device=x.device)
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(x[token_idx])
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer("k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False)

    def forward(self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)
        return k, v

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)
