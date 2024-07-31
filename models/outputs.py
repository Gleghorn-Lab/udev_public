import torch
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass


@dataclass
class MoEsmOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None
    r_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    pooler_output: torch.FloatTensor = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class AutoEncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    recon_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    z: Optional[torch.FloatTensor] = None
    reconstruction: Optional[torch.FloatTensor] = None
    bottleneck_cls: Optional[torch.FloatTensor] = None


@dataclass
class RotationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    rotation_matrix: Optional[torch.FloatTensor] = None


@dataclass
class LowMemMaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    full_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class GeneralModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None