import torch
import torch.nn as nn
import numpy as np
import inspect
import json
import pytorch_lightning as pl
from dataclasses import dataclass, asdict


@dataclass
class Config:
    def isolate(self, config):
        specifics = inspect.signature(config).parameters
        my_specifics = {k: v for k, v in asdict(self).items() if k in specifics}
        return config(**my_specifics)

    def to_json(self, filename):
        config = json.dumps(asdict(self), indent=2)
        with open(filename, 'w') as f:
            f.write(config)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            js = json.loads(f.read())
        config = cls(**js)
        return config


@dataclass
class trans_basic_block_Config_tmvec(Config):
    d_model: int = 1024
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 2048
    out_dim: int = 512
    dropout: float = 0.1
    activation: str = 'relu'
    # data params
    lr0: float = 0.0001
    warmup_steps: int = 300

    def build(self):
        return trans_basic_block_tmvec(self)


class trans_basic_block_tmvec(pl.LightningModule):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self, config: trans_basic_block_Config_tmvec):
        super().__init__()
        self.config = config

        # build encoder
        encoder_args = {k: v for k, v in asdict(config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters} 
        num_layers = config.num_layers

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = nn.Linear(self.config.d_model, self.config.out_dim)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, x, src_mask, src_key_padding_mask):
        x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        x = x.sum(dim=1) / lens.unsqueeze(1)
        x = self.dropout(x)
        x = self.mlp(x)
        return x

    
    def distance_loss_euclidean(self, output_seq1, output_seq2, tm_score):
        pdist_seq = nn.PairwiseDistance(p=2)
        dist_seq = pdist_seq(output_seq1, output_seq2)
        dist_tm = torch.cdist(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        return dist_tm

    def distance_loss_sigmoid(self, output_seq1, output_seq2, tm_score):
        dist_seq = output_seq1 - output_seq2
        dist_seq = torch.sigmoid(dist_seq).mean(1)
        dist_tm = torch.cdist(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        return dist_tm

    def distance_loss(self, output_seq1, output_seq2, tm_score):
        dist_seq = self.cos(output_seq1, output_seq2)  
        dist_tm = self.l1_loss(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0))
        return dist_tm

    def training_step(self, train_batch, batch_idx):
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = train_batch
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        loss = self.distance_loss(out_seq1, out_seq2, tm_score)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = val_batch
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        loss = self.distance_loss(out_seq1, out_seq2, tm_score)
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]


@dataclass
class trans_basic_block_Config_single(Config):
    d_model: int = 1024
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 2048
    out_dim: int = 512
    dropout: float = 0.1
    activation: str = 'relu'
    num_variables: int = 10 #9
    vocab: int = 20
    # data params
    lr0: float = 0.0001
    warmup_steps: int = 300
    p_bernoulli: float = .5
    
    def build(self):
        return trans_basic_block_single(self)


class trans_basic_block_single(pl.LightningModule):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self, config: trans_basic_block_Config_single):
        super().__init__()
        self.config = config

        #Encoding
        encoder_args = {k: v for k, v in asdict(config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters} 
        num_layers = config.num_layers
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #Linear and dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        #2 layer approach: 
        hidden_dim = self.config.d_model
        self.mlp_1 = nn.Linear(hidden_dim, self.config.out_dim)
        self.mlp_2 = nn.Linear(self.config.out_dim, self.config.out_dim)
        
        #Loss functions 
        self.trip_margin_loss = nn.TripletMarginLoss(margin=1.0, reduction='mean')#p=2)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.pdist = nn.PairwiseDistance(p=2)
        
    def forward(self, x_i, src_mask, src_key_padding_mask):
        enc_out = self.encoder(x_i, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        out = enc_out.sum(dim=1) / lens.unsqueeze(1)

        out = self.mlp_1(out)
        out = self.dropout(out)
        out = self.mlp_2(out)
        return out
    
    
    def triplet_margin_loss(self, output_seq1, output_seq2, output_seq3):
        loss = self.trip_margin_loss(output_seq1, output_seq2, output_seq3)
        return loss
    
    def distance_marginal_triplet(self, out_seq1, out_seq2, out_seq3, margin):
        d1 = self.pdist(out_seq1, out_seq2)
        d2 = self.pdist(out_seq1, out_seq3)
        zeros = torch.zeros(d1.shape).to(out_seq1)
        margin = margin.to(out_seq1)
        loss = torch.mean(torch.max(d1 - d2 + margin, zeros))

        return(loss)

    def distance_loss(self, output_seq1, output_seq2, output_seq3, margin):
        dist_seq1 = self.cos(output_seq1, output_seq2)
        dist_seq2 = self.cos(output_seq1, output_seq3)
        margin = margin.to(output_seq1)
        diff = dist_seq2 - dist_seq1
        dist_margin = self.l1_loss(diff.unsqueeze(0), margin.float().unsqueeze(0))
        
        return dist_margin

    def distance_loss2(self, output_seq1, output_seq2, output_seq3, margin):
        dist_seq1 = self.cos(output_seq1, output_seq2)
        dist_seq2 = self.cos(output_seq1, output_seq3)
        margin = margin.to(output_seq1)                                                                                                                    
        zeros = torch.zeros(dist_seq1.shape).to(output_seq1)
        loss = torch.mean(torch.max(dist_seq1 - dist_seq2 + margin, zeros))
        return loss
    
    def training_step(self, train_batch, batch_idx):
        #key_vars = train_batch['key']
        margins = torch.FloatTensor(train_batch['key'])

        #Get the ID embeddings
        sequence_1 = train_batch['id']
        pad_mask_1 = train_batch['id_padding']
        sequence_2 = train_batch['positive']
        pad_mask_2 = train_batch['positive_padding']
        sequence_3 = train_batch['negative']
        pad_mask_3 = train_batch['negative_padding']
        
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        out_seq3 = self.forward(sequence_3, src_mask=None, src_key_padding_mask=pad_mask_3)

        loss = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)
        
        self.log('train_loss', loss, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        #key_vars = val_batch['key']
        margins = torch.FloatTensor(val_batch['key'])
        
        #Get the ID embeddings
        sequence_1 = val_batch['id']
        pad_mask_1 = val_batch['id_padding']
        sequence_2 = val_batch['positive']
        pad_mask_2 = val_batch['positive_padding']
        sequence_3 = val_batch['negative']
        pad_mask_3 = val_batch['negative_padding']

        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        out_seq3 = self.forward(sequence_3, src_mask=None, src_key_padding_mask=pad_mask_3)        

        loss = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)
        
        self.log('val_loss', loss, sync_dist=True)

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        return [optimizer], [lr_scheduler]


@dataclass
class trans_basic_block_Config(Config):
    d_model: int = 512
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 2048
    out_dim: int = 512
    dropout: float = 0.1
    activation: str = 'relu'
    num_variables: int = 10 #9
    vocab: int = 20
    # data params
    lr0: float = 0.0001
    warmup_steps: int = 300
    p_bernoulli: float = .5
    
    def build(self):
        return trans_basic_block(self)


class trans_basic_block(pl.LightningModule):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self, config: trans_basic_block_Config):
        super().__init__()
        self.config = config

        #Encoding
        encoder_args = {k: v for k, v in asdict(config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters} 
        num_layers = config.num_layers
        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        #Linear and dropout
        self.dropout = nn.Dropout(self.config.dropout)

        # Define 1D convolutional layer
        
        #2 layer approach: 
        self.mlp_1 = nn.Linear(self.config.d_model, self.config.out_dim)
        self.mlp_2 = nn.Linear(self.config.out_dim, self.config.out_dim)
        
        #Loss functions 
        self.trip_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        
        #embedding lookup
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')
        
        self.pdist = nn.PairwiseDistance(p=2)

        ################## TM-Vec model
        vec_model_cpnt_tmvec = 'protein_vec_models/tm_vec_swiss_model_large.ckpt'
        vec_model_config_tmvec = 'protein_vec_models/tm_vec_swiss_model_large_params.json'
        
        #Load the model
        vec_model_config_tmvec = trans_basic_block_Config_tmvec.from_json(vec_model_config_tmvec)
        self.model_aspect_tmvec = trans_basic_block_tmvec.load_from_checkpoint(vec_model_cpnt_tmvec, config=vec_model_config_tmvec)
        for param in self.model_aspect_tmvec.parameters():
            param.requires_grad = False

        ################## PFam model
        vec_model_cpnt_pfam = 'protein_vec_models/aspect_vec_pfam.ckpt'
        vec_model_config_pfam = 'protein_vec_models/aspect_vec_pfam_params.json'
        #Load the model
        vec_model_config_pfam = trans_basic_block_Config_single.from_json(vec_model_config_pfam)
        self.model_aspect_pfam = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_pfam, config=vec_model_config_pfam)
        for param in self.model_aspect_pfam.parameters():
            param.requires_grad = False

        ################## GENE3D model
        vec_model_cpnt_gene3D = 'protein_vec_models/aspect_vec_gene3d.ckpt'
        vec_model_config_gene3D = 'protein_vec_models/aspect_vec_gene3d_params.json'
        #Load the model
        vec_model_config_gene3D = trans_basic_block_Config_single.from_json(vec_model_config_gene3D)
        self.model_aspect_gene3D = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_gene3D, config=vec_model_config_gene3D)
        for param in self.model_aspect_gene3D.parameters():
            param.requires_grad = False

        ################## EC model
        vec_model_cpnt_ec = 'protein_vec_models/aspect_vec_ec.ckpt'
        vec_model_config_ec = 'protein_vec_models/aspect_vec_ec_params.json'
        #Load the model
        vec_model_config_ec = trans_basic_block_Config_single.from_json(vec_model_config_ec)
        self.model_aspect_ec = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_ec, config=vec_model_config_ec)
        for param in self.model_aspect_ec.parameters():
            param.requires_grad = False

        ################## GO MFO model
        vec_model_cpnt_mfo = 'protein_vec_models/aspect_vec_go_mfo.ckpt'
        vec_model_config_mfo = 'protein_vec_models/aspect_vec_go_mfo_params.json'
        #Load the model
        vec_model_config_mfo = trans_basic_block_Config_single.from_json(vec_model_config_mfo)
        self.model_aspect_mfo = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_mfo, config=vec_model_config_mfo)
        for param in self.model_aspect_mfo.parameters():
            param.requires_grad = False

        ################## GO BPO model
        vec_model_cpnt_bpo = 'protein_vec_models/aspect_vec_go_bpo.ckpt'
        vec_model_config_bpo = 'protein_vec_models/aspect_vec_go_bpo_params.json'
        #Load the model 
        vec_model_config_bpo = trans_basic_block_Config_single.from_json(vec_model_config_bpo)
        self.model_aspect_bpo = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_bpo, config=vec_model_config_bpo)
        for param in self.model_aspect_bpo.parameters():
            param.requires_grad = False

        ################## GO CCO model
        vec_model_cpnt_cco = 'protein_vec_models/aspect_vec_go_cco.ckpt'
        vec_model_config_cco = 'protein_vec_models/aspect_vec_go_cco_params.json'
        #Load the model
        vec_model_config_cco = trans_basic_block_Config_single.from_json(vec_model_config_cco)
        self.model_aspect_cco = trans_basic_block_single.load_from_checkpoint(vec_model_cpnt_cco, config=vec_model_config_cco)
        for param in self.model_aspect_cco.parameters():
            param.requires_grad = False

            
    def forward(self, x_i, src_key_padding_mask):
        #embedding
        src_key_padding_mask = src_key_padding_mask.to(x_i)
        enc_out = self.encoder(x_i, mask=None, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        enc_out = enc_out.sum(dim=1) / lens.unsqueeze(1)
        out = self.mlp_1(enc_out)
        out = self.mlp_2(out)
        
        return out

    def distance_marginal_triplet(self, out_seq1, out_seq2, out_seq3, margin):
        d1 = self.pdist(out_seq1, out_seq2)
        d2 = self.pdist(out_seq1, out_seq3)
        zeros = torch.zeros(d1.shape).to(out_seq1)
        margin = margin.to(out_seq1)
        loss = torch.mean(torch.max(d1 - d2 + margin, zeros))
        return(loss)
    
    def triplet_margin_loss(self, output_seq1, output_seq2, output_seq3):
        loss = self.trip_margin_loss(output_seq1, output_seq2, output_seq3)
        return loss
    
    def distance_loss_tm_positive(self, output_seq1, output_seq2, tm_score):
        dist_seq = self.cos(output_seq1, output_seq2) 
        dist_tm = self.l1_loss(dist_seq.unsqueeze(0), tm_score.unsqueeze(0))
        return dist_tm

    def distance_loss_tm_difference(self, output_seq1, output_seq2, output_seq3, tm_score):
        dist_seq1 = self.cos(output_seq1, output_seq2)
        dist_seq2 = self.cos(output_seq1, output_seq3)
        difference = dist_seq2 - dist_seq1
        dist_tm = self.l1_loss(difference.unsqueeze(0), tm_score.unsqueeze(0))
        return dist_tm
    

    def make_matrix(self, sequence, pad_mask):
        pad_mask = pad_mask.to(sequence)
        aspect1 = self.model_aspect_tmvec(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect2 = self.model_aspect_pfam(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect3 = self.model_aspect_gene3D(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect4 = self.model_aspect_ec(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect5 = self.model_aspect_mfo(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect6 = self.model_aspect_bpo(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        aspect7 = self.model_aspect_cco(sequence, src_mask=None, src_key_padding_mask=pad_mask)[:,None,:]
        combine_aspects = torch.cat([aspect1, aspect2, aspect3, aspect4, aspect5, aspect6, aspect7], dim=1)
        return combine_aspects
        
    def training_step(self, train_batch, batch_idx):
        lookup_dict = {
            'nothing': 1,  'ENZYME': 2, 'PFAM':3, 'MFO':4, 'BPO':5, 'CCO':6,
            'TM':7,'GENE3D':8
        }
        
        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        
        sampled_keys = train_batch['key']
        margins = train_batch['margin']
        tm_type = train_batch['tm']
        tm_scores = train_batch['tm_scores']

        subset_sampled_keys = [sampled_keys[j].split(",") for j in range(len(sampled_keys))]
        masks = []
        for i in range(len(subset_sampled_keys)):
            mask = [all_cols[k] in subset_sampled_keys[i] for k in range(len(all_cols))]
            masks.append(mask)
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))
        

        #Get the ID embeddings
        sequence_1 = train_batch['id']
        pad_mask_1 = train_batch['id_padding']
        sequence_2 = train_batch['positive']
        pad_mask_2 = train_batch['positive_padding']
        sequence_3 = train_batch['negative']
        pad_mask_3 = train_batch['negative_padding']

        #Make Aspect matrices
        out_seq1 = self.make_matrix(sequence_1, pad_mask_1)
        out_seq2 = self.make_matrix(sequence_2, pad_mask_2)
        out_seq3 = self.make_matrix(sequence_3, pad_mask_3)

        #Forward pass
        out_seq1 = self.forward(out_seq1, masks)
        out_seq2 = self.forward(out_seq2, masks)
        out_seq3 = self.forward(out_seq3, masks)
                
        #Triplet loss
        loss_trip = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)

        #Positive TM loss
        loss_tm_positive = self.distance_loss_tm_positive(out_seq1, out_seq2, tm_scores)
        loss_positive_mask = torch.tensor([tm_type[i] == 'Positive' for i in range(len(tm_type))]).to(loss_tm_positive).to(bool)
        loss_tm_positive_fin = loss_tm_positive.masked_fill(loss_positive_mask, 0.0)

        #TM difference loss
        loss_tm_difference = self.distance_loss_tm_difference(out_seq1, out_seq2, out_seq3, tm_scores)
        loss_difference_mask = torch.tensor([tm_type[i] == 'Difference' for i in range(len(tm_type))]).to(loss_tm_difference).to(bool)
        loss_tm_difference_fin = loss_tm_difference.masked_fill(loss_difference_mask, 0.0)

        #Combined TM loss
        loss_part_2 = (loss_tm_positive_fin + loss_tm_difference_fin).mean()

        #complete loss
        loss = loss_trip + loss_part_2

        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])

        sampled_keys = val_batch['key']
        margins = val_batch['margin']
        tm_type = val_batch['tm']
        tm_scores = val_batch['tm_scores']

        subset_sampled_keys = [sampled_keys[j].split(",") for j in range(len(sampled_keys))]
        masks = []
        for i in range(len(subset_sampled_keys)):
            mask = [all_cols[k] in subset_sampled_keys[i] for k in range(len(all_cols))]
            masks.append(mask)
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))
        
        
        #Get the ID embeddings
        sequence_1 = val_batch['id']
        pad_mask_1 = val_batch['id_padding']
        sequence_2 = val_batch['positive']
        pad_mask_2 = val_batch['positive_padding']
        sequence_3 = val_batch['negative']
        pad_mask_3 = val_batch['negative_padding']

        out_seq1 = self.make_matrix(sequence_1, pad_mask_1)
        out_seq2 = self.make_matrix(sequence_2, pad_mask_2)
        out_seq3 = self.make_matrix(sequence_3, pad_mask_3)

        out_seq1 = self.forward(out_seq1, masks)
        out_seq2 = self.forward(out_seq2, masks)
        out_seq3 = self.forward(out_seq3, masks)

        #triplet loss
        loss_trip = self.distance_marginal_triplet(out_seq1, out_seq2, out_seq3, margins)

        #positive tm loss
        loss_tm_positive = self.distance_loss_tm_positive(out_seq1, out_seq2, tm_scores)
        loss_positive_mask = torch.tensor([tm_type[i] == 'Positive' for i in range(len(tm_type))]).to(loss_tm_positive).to(bool)
        loss_tm_positive_fin = loss_tm_positive.masked_fill(loss_positive_mask, 0.0)

        #difference tm loss
        loss_tm_difference = self.distance_loss_tm_difference(out_seq1, out_seq2, out_seq3, tm_scores)
        loss_difference_mask = torch.tensor([tm_type[i] == 'Difference' for i in range(len(tm_type))]).to(loss_tm_difference).to(bool)
        loss_tm_difference_fin = loss_tm_difference.masked_fill(loss_difference_mask, 0.0)

        #complete loss
        loss_part_2 = (loss_tm_positive_fin + loss_tm_difference_fin).mean()
        #complete loss
        loss = loss_trip + loss_part_2
        
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
        return [optimizer], [lr_scheduler]



class BasicConfig(trans_basic_block_Config):
    @classmethod
    def from_huggingface(cls, prefix, proteinvec_config):
        current_attributes = cls.__dataclass_fields__.keys()
        filtered_attributes = {k[len(prefix)+1:]: v for k, v in proteinvec_config.__dict__.items() if k.startswith(prefix)}
        config_dict = {k: v for k, v in filtered_attributes.items() if k in current_attributes}
        config = cls(**config_dict)
        return config


class TmConfig(trans_basic_block_Config_tmvec):
    @classmethod
    def from_huggingface(cls, prefix, proteinvec_config):
        current_attributes = cls.__dataclass_fields__.keys()
        filtered_attributes = {k[len(prefix)+1:]: v for k, v in proteinvec_config.__dict__.items() if k.startswith(prefix)}
        config_dict = {k: v for k, v in filtered_attributes.items() if k in current_attributes}
        config = cls(**config_dict)
        return config
    

class SingleConfig(trans_basic_block_Config_single):
    @classmethod
    def from_huggingface(cls, prefix, proteinvec_config):
        current_attributes = cls.__dataclass_fields__.keys()
        filtered_attributes = {k[len(prefix)+1:]: v for k, v in proteinvec_config.__dict__.items() if k.startswith(prefix)}
        config_dict = {k: v for k, v in filtered_attributes.items() if k in current_attributes}
        config = cls(**config_dict)
        return config


class HF_trans_basic_block(trans_basic_block):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        self.config = config

        encoder_config = BasicConfig.from_huggingface(prefix='vec', proteinvec_config=config)
        encoder_args = {k: v for k, v in asdict(encoder_config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters}    
        self.dropout = nn.Dropout(encoder_config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_config.num_layers)
        self.mlp_1 = nn.Linear(encoder_config.d_model, encoder_config.out_dim)
        self.mlp_2 = nn.Linear(encoder_config.out_dim, encoder_config.out_dim)
        
        self.trip_margin_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.pdist = nn.PairwiseDistance(p=2)

        self.model_aspect_tmvec = trans_basic_block_tmvec(TmConfig.from_huggingface('tm', config))
        self.model_aspect_pfam = trans_basic_block_single(SingleConfig.from_huggingface('pfam', config))
        self.model_aspect_gene3D = trans_basic_block_single(SingleConfig.from_huggingface('gene3d', config))
        self.model_aspect_ec = trans_basic_block_single(SingleConfig.from_huggingface('ec', config))
        self.model_aspect_mfo = trans_basic_block_single(SingleConfig.from_huggingface('mf', config))
        self.model_aspect_bpo = trans_basic_block_single(SingleConfig.from_huggingface('bp', config))
        self.model_aspect_cco = trans_basic_block_single(SingleConfig.from_huggingface('cc', config))
