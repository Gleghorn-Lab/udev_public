import torch.nn as nn
import torch.nn.functional as F
from functools import partial


Linear = partial(nn.Linear, bias=False)


class VanillaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        ffn_dim = config.intermediate_size
        hidden_size = config.hidden_size
        self.w1 = Linear(hidden_size, ffn_dim)
        self.w2 = Linear(ffn_dim, hidden_size)

    def forward(self, hidden_states):
        return self.w2(F.silu(self.w1(hidden_states)))


# Adapted from https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/models/mixtral/modeling_mixtral.py
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        ffn_dim = config.intermediate_size
        hidden_size = config.hidden_size
        self.w1 = Linear(hidden_size, ffn_dim)
        self.w2 = Linear(ffn_dim, hidden_size)
        self.w3 = Linear(hidden_size, ffn_dim)

    def forward(self, hidden_states):
        current_hidden_states = F.silu(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
