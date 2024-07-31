import transformers.models.convbert as c_bert
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

"""
Base model that consists of ConvBert layer - from ANKH repo

Args:
    hidden_size: Dimension of the input embeddings.
    nhead: Integer specifying the number of heads for the `ConvBert` model.
    hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
    nlayers: Integer specifying the number of layers for the `ConvBert` model.
    kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
    dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
    pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
"""

def init_weights(self):
    initrange = 0.1
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)

    def forward(self, x):
        out = self.global_avg_pool1d(x)
        return out


class BaseModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        nhead: int,
        intermediate_size: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = None,
    ):

        super().__init__()

        self.model_type = "Transformer"
        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=hidden_size,
            num_attention_heads=nhead,
            intermediate_size=intermediate_size,
            conv_kernel_size=kernel_size,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=dropout,
        )

        self.transformer_encoder = nn.ModuleList(
            [c_bert.ConvBertLayer(encoder_layers_Config) for _ in range(num_layers)]
        )

        if pooling is not None:
            if pooling.lower() in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling.lower() == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(
                    f"Expected pooling to be [`avg`, `max`]. Recieved: {pooling}"
                )

    def convbert_forward(self, x):
        for convbert_layer in self.transformer_encoder:
            x = convbert_layer(x)[0]
        return x


class ConvBertForMultiClassClassification(BaseModule):
    def __init__(self, cfg):
        super(ConvBertForMultiClassClassification, self).__init__(
            hidden_size=cfg.hidden_size,
            nhead=cfg.nhead,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=1,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.model_type = "Transformer"
        self.num_labels = cfg.num_labels
        self.decoder = nn.Linear(cfg.hidden_size, cfg.num_labels)
        init_weights(self)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ConvBertForRegression(BaseModule):
    def __init__(self, cfg):
        if cfg.pooling is None:
            raise ValueError(
                '`pooling` cannot be `None` in a regression task. Expected ["avg", "max"].'
            )

        super(ConvBertForRegression, self).__init__(
            hidden_size=cfg.hidden_size,
            nhead=cfg.nhead,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=1,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.decoder = nn.Linear(cfg.hidden_size, 1)
        init_weights(self)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            # Ensure that logits and labels have the same size before computing loss
            loss = F.mse_loss(logits.flatten(), labels.flatten())
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ConvBertForMultiLabelClassification(BaseModule):
    def __init__(self, cfg):
        super(ConvBertForMultiLabelClassification, self).__init__(
            hidden_size=cfg.hidden_size,
            nhead=cfg.nhead,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=1,
            num_layers=cfg.num_layers,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            pooling=cfg.pooling,
        )

        self.model_type = "Transformer"
        self.num_labels = cfg.num_labels
        self.decoder = nn.Linear(cfg.hidden_size, cfg.num_labels)
        init_weights(self)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class ConvBertForMatrixRep(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_type = "Transformer"
        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=config.hidden_dim,
            num_attention_heads=config.nhead,
            intermediate_size=config.intermediate_dim,
            conv_kernel_size=config.kernel_size,
            num_hidden_layers=1,
            hidden_dropout_prob=config.dropout,
        )

        self.transformer_encoder = nn.ModuleList(
            [c_bert.ConvBertLayer(encoder_layers_Config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x, attention_mask=None):  # (B, L, d)
        for convbert_layer in self.transformer_encoder:
            x = convbert_layer(x, attention_mask=attention_mask)[0]
        return x  # (B, L, d)
