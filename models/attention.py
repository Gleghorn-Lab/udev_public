import math
import torch.nn as nn
from typing import Optional
from functools import partial
from transformers.models.mixtral.modeling_mixtral import MixtralAttention, MixtralFlashAttention2, apply_rotary_pos_emb
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from .embedding_layers import RotaryEmbedding, VecToPatch, PatchToVec


Linear = partial(nn.Linear, bias=False)


class SelfAttention(MixtralAttention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class SelfFlashAttention(MixtralFlashAttention2):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, self.hidden_siz)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()