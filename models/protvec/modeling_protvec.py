import torch
import torch.nn as nn
import numpy as np
import inspect
import re
import pytorch_lightning as pl
from dataclasses import asdict
from transformers import PreTrainedModel, T5EncoderModel, T5Config, T5Tokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from .config_protvec import ProteinVecConfig
from .blocks import *


class ProteinVec(PreTrainedModel):
    def __init__(self, config: ProteinVecConfig):
        super().__init__(config)
        self.config = config

        self.t5 = T5EncoderModel(config=T5Config.from_pretrained('lhallee/prot_t5_enc'))
        self.tokenizer = T5Tokenizer.from_pretrained('lhallee/prot_t5_enc')
        self.moe = HF_trans_basic_block(config)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.contrastive_loss = nn.TripletMarginLoss()
        self.aspect_to_keys_dict = {
            'EC': ['ENZYME'],
            'MF': ['MFO'],
            'BP': ['BPO'],
            'CC': ['CCO'],
            'IP': ['PFAM'],
            '3D': ['GENE3D'],
            'ALL': ['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'] 
        }
        self.all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        self.inference_aspect = config.inference_aspect
        self.inference_mask = self.get_mask(self.inference_aspect)

    def to_eval(self):
        self.t5 = self.t5.eval()
        self.moe = self.moe.eval()
    
    def to_half(self):
        self.t5 = self.t5.half()
        self.moe = self.moe.half()

    def get_mask(self, aspect):
        sampled_keys = np.array(self.aspect_to_keys_dict[aspect])
        masks = [self.all_cols[k] in sampled_keys for k in range(len(self.all_cols))]
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]
        return masks

    def featurize_prottrans(self, sequences):
        sequences = [(" ".join(sequences[i])) for i in range(len(sequences))]
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        ids = self.tokenizer.batch_encode_plus(sequences,
                                          add_special_tokens=True,
                                          padding=True,
                                          max_length=1024,
                                          truncation=True)
        input_ids = torch.tensor(ids['input_ids']).to(self.dev)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.dev)
        with torch.no_grad():
            embedding = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.float()
        features = [] 
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            features.append(seq_emd)
        prottrans_embedding = torch.tensor(features[0])
        prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0)
        return(prottrans_embedding)

    def embed_vec(self, prottrans_embedding, masks):
        padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(self.dev)
        out_seq = self.moe.make_matrix(prottrans_embedding, padding)
        vec_embedding = self.moe(out_seq, masks)
        return vec_embedding

    def embed(self, seq):
        protrans_sequence = self.featurize_prottrans([seq])
        return self.embed_vec(protrans_sequence, self.inference_mask)

    def forward(self, p_seqs, a_seqs, n_seqs, aspect):
        p = self.embed(p_seqs, aspect=aspect)
        a = self.embed(a_seqs, aspect=aspect)
        n = self.embed(n_seqs, aspect=aspect)

        loss = self.contrastive_loss(p, a, n)
        logits = (p, a, n)

        return SequenceClassifierOutput(
            logits=logits,
            loss=loss
        )
