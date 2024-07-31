import torch
import torch.nn as nn
import ankh
from functools import partial
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from ..heads import LanguageModelingHead
from ..convbert.modeling_convbert import ConvBertForMatrixRep
from ...losses.contrastive import diff_loss, MNR_loss, space_loss, clip_loss
from ..esm.custom_esm import CustomEsmForMaskedLM



# Contrastive Annotation Modeling for Proteins
class CAMP_Objectives(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.plm = AutoModel.from_pretrained(config.plm_path, token=config.token)
        self.nlp = AutoModel.from_pretrained(config.nlp_path, token=config.token)
        self.vocab_size = self.plm.config.vocab_size
        plm_dim = self.plm.config.hidden_size
        nlp_dim = self.nlp.config.hidden_size
        out_dim = config.out_dim
        half_dim = plm_dim//2
        if config.bias:
            Linear = partial(nn.Linear, bias=True)
        else:
            Linear = partial(nn.Linear, bias=False)

        self.plm_reduce = Linear(plm_dim, half_dim)
        self.nlp_reduce = Linear(nlp_dim, half_dim)
        self.plm_proj = Linear(half_dim, out_dim)
        self.nlp_proj = Linear(half_dim, out_dim)

        config.hidden_dim = half_dim
        config.intermediate_dim = half_dim * 4
        self.plm_convbert = ConvBertForMatrixRep(config)
        self.nlp_convbert = ConvBertForMatrixRep(config)

        self.mnr = config.mnr
        self.diff = config.diff
        self.latent = config.latent
        self.mlm = config.mlm
        if self.latent:
            self.latent_loss = nn.SmoothL1Loss()
        if self.mlm:
            self.lm_loss = nn.CrossEntropyLoss()
            self.lm_head = LanguageModelingHead(half_dim, self.vocab_size)
        if self.diff:
            self.diff_loss = diff_loss
        if self.mnr:
            self.mnr_loss = MNR_loss
        if self.space:
            self.space_loss = space_loss
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def freeze(self):
        for param in self.plm.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = False
        for param in self.nlp.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.plm.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True
        for param in self.nlp.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True
    
    def pool(self, hidden_state, attention_mask):
        non_pad_mask = attention_mask.bool()
        hidden_state = (hidden_state * non_pad_mask.unsqueeze(-1)).sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True)
        return hidden_state

    def plm_forward(self, input_ids, attention_mask):
            return self.plm_convbert(self.plm_reduce(self.plm(input_ids, attention_mask=attention_mask).last_hidden_state.float()))

    def forward(self, plm_tok, nlp_tok, labels=None):
        losses = []
        plm_mask, nlp_mask = plm_tok['attention_mask'], nlp_tok['attention_mask']

        if self.latent:
            plm_pred = self.plm_forward(plm_tok['input_ids'], plm_mask)
            plm_target = self.plm_forward(plm_tok['original_ids'], plm_mask)
            losses.append(self.latent_loss(plm_pred, plm_target))

        if self.mlm:
            mlm_labels = plm_tok['labels']
            plm_target = self.plm_forward(plm_tok['input_ids'], plm_mask)
            lm_logits = self.lm_head(plm_target)
            losses.append(self.lm_loss(lm_logits.view(-1, self.vocab_size), mlm_labels.view(-1))) # mlm

        if not self.latent and not self.mlm:
            plm_target = self.plm_forward(plm_tok['input_ids'], plm_mask)

        plm_vecs = self.plm_proj(self.pool(plm_target, plm_mask))

        nlp_rep = self.nlp_convbert(self.nlp_reduce(self.nlp(**nlp_tok).last_hidden_state.float()))
        nlp_vecs = self.nlp_proj(self.pool(nlp_rep, nlp_mask))

        if self.diff:
            losses.append(self.diff_loss(plm_vecs, nlp_vecs))

        if self.mnr:
            losses.append(self.mnr_loss(plm_vecs, nlp_vecs))

        if self.space:
            losses.append(self.space_loss(plm_vecs, nlp_vecs))

        loss = sum(losses)
        logits = (plm_vecs, nlp_vecs)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class CAMP(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.plm = AutoModel.from_pretrained(config.plm_path, token=config.token)
        self.nlp = AutoModel.from_pretrained(config.nlp_path, token=config.token)
        self.vocab_size = self.plm.config.vocab_size
        plm_dim = self.plm.config.hidden_size
        nlp_dim = self.nlp.config.hidden_size
        out_dim = config.out_dim
        half_dim = plm_dim//2
        if config.bias:
            Linear = partial(nn.Linear, bias=True)
        else:
            Linear = partial(nn.Linear, bias=False)

        self.plm_reduce = Linear(plm_dim, half_dim)
        self.nlp_reduce = Linear(nlp_dim, half_dim)
        self.plm_proj = Linear(half_dim, out_dim)
        self.nlp_proj = Linear(half_dim, out_dim)

        config.hidden_dim = half_dim
        config.intermediate_dim = config.intermediate_dim
        self.plm_convbert = ConvBertForMatrixRep(config)
        self.nlp_convbert = ConvBertForMatrixRep(config)

        self.criterion = space_loss
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def freeze(self):
        for param in self.plm.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = False
        for param in self.nlp.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.plm.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True
        for param in self.nlp.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True
    
    def pool(self, hidden_state, attention_mask):
        non_pad_mask = attention_mask.bool()
        hidden_state = (hidden_state * non_pad_mask.unsqueeze(-1)).sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True)
        return hidden_state

    def forward(self, plm_tok, nlp_tok, labels=None): # labels for compute_metrics() during eval
        plm_mask, nlp_mask = plm_tok['attention_mask'], nlp_tok['attention_mask']

        plm_rep = self.plm_convbert(self.plm_reduce(self.plm(**plm_tok).last_hidden_state.float()))
        plm_vecs = self.plm_proj(self.pool(plm_rep, plm_mask))

        nlp_rep = self.nlp_convbert(self.nlp_reduce(self.nlp(**nlp_tok).last_hidden_state.float()))
        nlp_vecs = self.nlp_proj(self.pool(nlp_rep, nlp_mask))

        loss = self.criterion(plm_vecs, nlp_vecs)

        logits = (plm_vecs, nlp_vecs)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


def ProjectionHead(
    hidden_dim: int,
    output_dim: int,
    bias: bool = True, # we use bias for pooled reps
) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(hidden_dim, bias=bias),
        nn.Linear(hidden_dim, hidden_dim, bias=bias),
        nn.SiLU(),
        nn.LayerNorm(hidden_dim, bias=bias),
        nn.Linear(hidden_dim, output_dim, bias=bias),
    )


class CAMPfinal(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.plm, self.tokenizer = ankh.load_base_model(generation=False) # model, tokenizer
        self.nlp = CustomEsmForMaskedLM.from_pretrained(config.nlp_path, token=config.token)
        self.vocab_size = self.plm.config.vocab_size
        plm_dim = self.plm.config.d_model
        nlp_dim = self.nlp.config.hidden_size
        out_dim = config.out_dim
        half_dim = plm_dim//2

        self.plm_reduce = ProjectionHead(plm_dim, half_dim)
        self.nlp_reduce = ProjectionHead(nlp_dim, half_dim)
        self.plm_proj = ProjectionHead(half_dim, out_dim)
        self.nlp_proj = ProjectionHead(half_dim, out_dim)

        config.hidden_dim = half_dim
        config.intermediate_dim = config.intermediate_dim
        self.plm_convbert = ConvBertForMatrixRep(config)
        self.nlp_convbert = ConvBertForMatrixRep(config)

        self.criterion = clip_loss
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def freeze(self):
        for param in self.plm.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = False
        for param in self.nlp.parameters():
            param.data = param.data.to(torch.bfloat16)
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.plm.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True
        for param in self.nlp.parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = True

    def pool(self, hidden_state, attention_mask):
        non_pad_mask = attention_mask.bool()
        hidden_state = (hidden_state * non_pad_mask.unsqueeze(-1)).sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True)
        return hidden_state

    def forward(self, plm_tok, nlp_tok, labels=None): # labels for compute_metrics() during eval
        plm_mask, nlp_mask = plm_tok['attention_mask'], nlp_tok['attention_mask']

        plm_rep = self.plm_convbert(self.plm_reduce(self.plm(**plm_tok).last_hidden_state.float()))
        plm_vecs = self.plm_proj(self.pool(plm_rep, plm_mask))

        nlp_rep = self.nlp_convbert(self.nlp_reduce(self.nlp(**nlp_tok).last_hidden_state.float()))
        nlp_vecs = self.nlp_proj(self.pool(nlp_rep, nlp_mask))

        loss = self.criterion(plm_vecs, nlp_vecs)

        logits = (plm_vecs, nlp_vecs)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )