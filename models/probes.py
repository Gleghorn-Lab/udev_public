from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from .convbert.modeling_convbert import *
from ..losses.get_loss_fct import get_loss_fct


class LinearProbe(nn.Module):
    def __init__(self, cfg, input_dim=768, task_type='binary', num_labels=2):
        super().__init__()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.num_layers = cfg.num_layers
        self.input_layer = nn.Linear(input_dim, cfg.hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            self.hidden_layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.output_layer = nn.Linear(cfg.hidden_dim, num_labels)
        self.loss_fct = get_loss_fct(task_type)

    def forward(self, embeddings, labels=None):
        embeddings = self.gelu(self.input_layer(embeddings))
        for i in range(self.num_layers):
            embeddings = self.dropout(self.gelu(self.hidden_layers[i](embeddings)))
        logits = self.output_layer(embeddings)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class ConvBertProbe(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.task_type == 'singlelabel':
            self.model = ConvBertForMultiClassClassification(cfg)
        elif cfg.task_type == 'multilabel':
            self.model = ConvBertForMultiLabelClassification(cfg)
        elif cfg.task_type == 'regression':
            self.model = ConvBertForRegression(cfg)
        else:
            print('You did not pass a correct task type:\n binary , multiclass , multilabel , regression')
            print('You passed: ', cfg.task_type)
        self.encoder = nn.Linear(cfg.input_size, cfg.hidden_size)

    def forward(self, embeddings, labels=None):
        embeddings = self.encoder(embeddings)
        out = self.model(embeddings, labels)
        return out


class BertProbe(nn.Module):
    def __init__(self, cfg, input_dim=768, task_type='binary', num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.proj = nn.Linear(input_dim, cfg.hidden_dim)
        self.bert_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.intermediate_dim,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True
        )
        self.bert = nn.TransformerEncoder(
            encoder_layer=self.bert_layer,
            num_layers=cfg.num_layers
        )
        self.head = nn.Linear(cfg.hidden_dim, num_labels)
        self.loss_fct = get_loss_fct(task_type)
    
    def forward(self, embeddings, labels=None):
        embeddings = self.proj(embeddings)
        last_hidden_state = self.bert(embeddings)
        logits = self.head(last_hidden_state)

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
