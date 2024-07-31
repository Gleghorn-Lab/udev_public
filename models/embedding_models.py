import torch
from transformers import T5EncoderModel, EsmModel, AutoTokenizer, PreTrainedModel
from .protvec.modeling_protvec import ProteinVec
from .jesm.modeling_jesm import JESM
from .jesm.modeling_jesm_v2 import JESMv2
from .camp.modeling_camp import CAMP, CAMPfinal
from .camp.modeling_campv2 import CAMPv2
from .esm.modeling_esm3 import ESM3Custom, ESMProtein


class RandomEmbedder:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def embed(self, seq, full=False):
        if full:
            seq_len = len(seq)
            return torch.randn(1, seq_len, self.embedding_dim)
        else:
            return torch.randn(1, self.embedding_dim)


class SelfiesEmbedder(torch.nn.Module):
    def __init__(self, selformer, tokenizer):
        super().__init__()
        self.config = selformer.config
        self.selfomer = selformer
        self.tokenizer = tokenizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        tok = self.tokenizer(seq, padding=False, return_tensors='pt')
        tok = self.move_to_device(tok)
        emb = self.selfomer(**tok).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)

    def embed_train(self, input_ids, attention_mask, full=False):
        emb = self.selfomer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        if full:
            return emb
        else:
            return emb.mean(dim=1)


class AnkhForEmbedding(T5EncoderModel):
    def __init__(self, ankh, tokenizer):
        super().__init__(ankh.config)
        self.ankh = ankh
        self.tokenizer = tokenizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        tok = self.tokenizer(seq, padding=False, return_tensors='pt')
        tok = self.move_to_device(tok)
        emb = self.ankh(**tok).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)
    
    def embed_train(self, input_ids, attention_mask, full=False):
        emb = self.ankh(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)


class EsmForEmbedding(torch.nn.Module):
    def __init__(self, esm, tokenizer):
        super().__init__()
        self.esm = esm
        self.tokenizer = tokenizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        tok = self.tokenizer(seq, padding=False, return_tensors='pt')
        tok = self.move_to_device(tok)
        emb = self.esm(**tok).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)
    
    def embed_train(self, input_ids, attention_mask, full=False):
        emb = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)
        

class BertForEmbedding(torch.nn.Module):
    def __init__(self, bert, tokenizer):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        seq = ' '.join(list(seq))
        tok = self.tokenizer(seq, padding=False, return_tensors='pt')
        tok = self.move_to_device(tok)
        emb = self.bert(**tok).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)
    
    def embed_train(self, input_ids, attention_mask, full=False):
        emb = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)
        

class Esm3ForEmbedding(ESM3Custom):
    def __init__(self, config):
        super().__init__(config)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def embed(self, seq, full=False):
        tokens = ESMProtein(sequence=seq) # tokenize
        tokens = self.esm.encode(tokens) # to tensor format
        out = self.esm(sequence_tokens = tokens.sequence.unsqueeze(0).to(self._device)) # needs to be batched
        emb = out.embeddings.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)


class ProteinVecForEmbedding(ProteinVec):
    def __init__(self, config):
        super().__init__(config)
    
    def embed(self, seq, full=False):
        protrans_sequence = self.featurize_prottrans([seq])
        return self.embed_vec(protrans_sequence, self.inference_mask)


class JesmForEmbedding(JESM):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        tok = self.tokenizer(seq, padding=False, return_tensors='pt')
        tok = self.move_to_device(tok)
        emb = self.encoder(**tok).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = emb.mean(dim=1) # mean pooling
            return emb # (1, d)


class CampForEmbedding(CAMP):
    def __init__(self, config):
        super().__init__(config)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_path)

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        plm_tok = self.tokenizer(seq, padding=False, return_tensors='pt') # still returns attention mask
        plm_tok = self.move_to_device(plm_tok)
        plm_rep = self.plm(**plm_tok).last_hidden_state.float()
        plm_rep = self.plm_reduce(plm_rep)
        plm_rep = self.plm_convbert(plm_rep)
        if full:
            return plm_rep
        else:
            return self.plm_proj(self.pool(plm_rep, plm_tok['attention_mask']))

    def embed_train(self, input_ids, attention_mask, full=False):
        plm_rep = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        plm_rep = self.plm_reduce(plm_rep)
        plm_rep = self.plm_convbert(plm_rep)
        if full:
            return plm_rep
        else:
            return self.plm_proj(self.pool(plm_rep, attention_mask))


class CampFinalForEmbedding(CAMPfinal):
    def __init__(self, config):
        super().__init__(config)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.plm_path)

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        plm_tok = self.tokenizer(seq, padding=False, return_tensors='pt') # still returns attention mask
        plm_tok = self.move_to_device(plm_tok)
        plm_rep = self.plm(**plm_tok).last_hidden_state.float()
        plm_rep = self.plm_reduce(plm_rep)
        plm_rep = self.plm_convbert(plm_rep)
        if full:
            return plm_rep
        else:
            return self.plm_proj(self.pool(plm_rep, plm_tok['attention_mask']))

    def embed_train(self, input_ids, attention_mask, full=False):
        plm_rep = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        plm_rep = self.plm_reduce(plm_rep)
        plm_rep = self.plm_convbert(plm_rep)
        if full:
            return plm_rep
        else:
            return self.plm_proj(self.pool(plm_rep, attention_mask))



class Campv2ForEmbedding(torch.nn.Module):
    def __init__(self, camp):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.camp = camp

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        plm_tok = self.camp.tokenizer(seq, padding=False, return_tensors='pt') # still returns attention mask
        plm_tok = self.move_to_device(plm_tok)
        out = self.camp.target_encoder(**plm_tok)
        if full:
            return out.last_hidden_state # (B, L, c)
        else:
            return self.camp.target_pool_proj(out.pooler_output) # (B, c)
        

class Campv3ForEmbedding(torch.nn.Module):
    def __init__(self, camp, base_tokenizer):
        super().__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_tokenizer = base_tokenizer
        self.camp = camp

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        context_tok = self.camp.tokenizer(seq, padding=False, return_tensors='pt') # still returns attention mask
        context_tok = self.move_to_device(context_tok)
        out = self.camp.context_encoder(**context_tok).last_hidden_state
        if full:
            return out
        else:

            return out.mean(dim=1)



class Jesm2ForEmbedding(JESMv2):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move_to_device(self, tokenized):
        return {k:v.to(self._device) for k, v in tokenized.items()}

    def embed(self, seq, full=False):
        tok = self.tokenizer(seq, padding=False, return_tensors='pt')
        tok = self.move_to_device(tok)
        emb = self.context_encoder(**tok).last_hidden_state.float()
        if full:
            return emb # (1, L, d)
        else:
            emb = self.context_pooler(emb[:, 0]) # cls pooling
            return emb # (1, d)


class HybridModel(PreTrainedModel):
    def __init__(self, embedding_model, probe, full=False):
        super().__init__(embedding_model.config)
        self.embedding_model = embedding_model
        self.probe = probe
        self.full = full

    def forward(self, input_ids, attention_mask=None, labels=None):
        emb = self.embedding_model.embed_train(input_ids, attention_mask, full=self.full)
        return self.probe(emb, labels)


def get_plm(args, plm_path, model_type, aspect=None, eval=True, return_tokenizer=False, subfolder=''):
    print('\n-----Load PLM-----\n')
    print(f'PLM: {plm_path}, Model: {model_type}')
    tokenizer = None
    if model_type.lower() == 'esm':
        model = EsmModel.from_pretrained(plm_path, token=args.token)
        tokenizer = AutoTokenizer.from_pretrained(plm_path, token=args.token)
        plm = EsmForEmbedding(model, tokenizer)
        if eval:
            plm = plm.eval()
    if model_type.lower() == 'bert':
        from transformers import BertModel
        model = BertModel.from_pretrained(plm_path, token=args.token)
        tokenizer = AutoTokenizer.from_pretrained(plm_path, token=args.token)
        plm = EsmForEmbedding(model, tokenizer)
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'ankh':
        import ankh
        if plm_path.lower() == 'ankh-base':
            model, tokenizer = ankh.load_base_model(generation=False)
        elif plm_path.lower() == 'ankh-large':
            model, tokenizer = ankh.load_large_model(generation=False)
        plm = AnkhForEmbedding(model, tokenizer)
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'asm':
        from .esm.custom_esm import CustomEsmForMaskedLM
        model = CustomEsmForMaskedLM.from_pretrained(plm_path, token=args.token)
        tokenizer = AutoTokenizer.from_pretrained(plm_path, token=args.token)
        plm = EsmForEmbedding(model, tokenizer)
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'camp':
        from .camp.config_camp import CAMPConfig
        config = CAMPConfig.from_pretrained(plm_path, token=args.token, subfolder=subfolder)
        config.token = args.token
        plm = CampForEmbedding.from_pretrained(plm_path, config=config, token=args.token, subfolder=subfolder)
        plm.freeze()
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'camp_final':
        from .camp.config_camp import CAMPConfig
        config = CAMPConfig.from_pretrained(plm_path, token=args.token, subfolder=subfolder)
        config.token = args.token
        plm = CampFinalForEmbedding.from_pretrained(plm_path, config=config, token=args.token, subfolder=subfolder)
        plm.freeze()
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'campv2':
        from .camp.config_campv2 import CAMPv2Config
        from .camp.modeling_campv2 import CAMPv2
        config = CAMPv2Config.from_pretrained(plm_path, token=args.token, subfolder=subfolder)
        config.token = args.token
        model = CAMPv2.from_pretrained(plm_path, config=config, token=args.token, subfolder=subfolder)
        plm = Campv2ForEmbedding(model)
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'campv3':
        from .camp.config_campv3 import CAMPv3Config
        from .camp.modeling_campv3 import CAMPv3
        config = CAMPv3Config.from_pretrained(plm_path, token=args.token, subfolder=subfolder)
        config.token = args.token
        model = CAMPv3.from_pretrained(plm_path, config=config, token=args.token, subfolder=subfolder)
        base_tokenizer = AutoTokenizer.from_pretrained('ElnaggarLab/ankh-base', token=args.token)
        plm = Campv3ForEmbedding(model, base_tokenizer)
        if eval:
            plm = plm.eval()
    elif model_type.lower() == 'protvec':
        from .protvec.config_protvec import ProteinVecConfig
        config = ProteinVecConfig()
        config.inference_aspect = 'ALL'
        plm = ProteinVecForEmbedding.from_pretrained('lhallee/ProteinVec', config=config, token=args.token)
        if eval:
            plm.to_eval()
    elif model_type.lower() == 'aspectvec':
        print(f'Aspect: {aspect}')
        from .protvec.config_protvec import ProteinVecConfig
        config = ProteinVecConfig()
        plm = ProteinVecForEmbedding.from_pretrained('lhallee/ProteinVec', config=config, token=args.token)
        if eval:
            plm.to_eval()
    elif model_type.lower() == 'esm3':
        from .esm.modeling_esm3 import ESM3Config
        config = ESM3Config()
        plm = Esm3ForEmbedding.from_pretrained('GleghornLab/esm3', config=config, token=args.token)
        if eval:
            plm.eval()
    elif model_type.lower() == 'selformer':
        from transformers import AutoModel
        selformer = AutoModel.from_pretrained('HUBioDataLab/SELFormer')
        tokenizer = AutoTokenizer.from_pretrained('HUBioDataLab/SELFormer')
        plm = SelfiesEmbedder(selformer, tokenizer)
        if eval:
            plm.eval()
    elif model_type.lower() == 'random_esm':
        from transformers import EsmConfig
        config = EsmConfig.from_pretrained(plm_path, token=args.token)
        model = EsmModel(config)
        tokenizer = AutoTokenizer.from_pretrained(plm_path, token=args.token)
        plm = EsmForEmbedding(model, tokenizer)
        if eval:
            plm.eval()
    elif model_type.lower() == 'random':
        from transformers import EsmConfig
        config = EsmConfig.from_pretrained(plm_path, token=args.token)
        hidden_size = config.hidden_size
        plm = RandomEmbedder(hidden_size)
    if return_tokenizer:
        if tokenizer == None:
            tokenizer = AutoTokenizer.from_pretrained(plm_path, token=args.token)
        return plm, tokenizer
    else:
        return plm
