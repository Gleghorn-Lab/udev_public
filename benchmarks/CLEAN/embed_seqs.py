import torch
import sqlite3
import numpy as np
import pandas as pd
from glob import glob
from transformers import EsmTokenizer
from current_model import MoEsmVec
from tqdm.auto import tqdm


def prepare_embed_standard_model(model, tokenizer, device, full=False, pooling='mean', max_length=None):
    def embed_standard_model(seqs):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(seqs, desc='Embedding batch'):
                ids = tokenizer(seq,
                        add_special_tokens=True,
                        padding=False,
                        return_token_type_ids=False,
                        return_tensors='pt').input_ids.to(device)
                output = model(ids)
                try:
                    emb = output.last_hidden_state.float()
                except:
                    emb = output.hidden_states[-1].float()
                if full:
                    if emb.size(1) < max_length:
                        padding_needed = max_length - emb.size(1)
                        emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed, 0, 0), value=0)
                    else:
                        emb = emb[:, :max_length, :]
                else:
                    if pooling == 'cls':
                        emb = emb[:, 0, :]
                    elif pooling == 'mean':
                        emb = torch.mean(emb, dim=1, keepdim=False)
                    else:
                        emb = torch.max(emb, dim=1, keepdim=False)[0]
                embeddings.append(emb.detach().cpu().numpy())
        return embeddings
    return embed_standard_model


def prepare_embed_double_model(model, tokenizer, device):
    def embed_double_model(seqs):
        embeddings = []
        with torch.no_grad():
            for seq in tqdm(seqs, desc='Embedding batch'):
                toks = tokenizer(seq,
                                add_special_tokens=True,
                                padding=False,
                                return_token_type_ids=False,
                                return_tensors='pt')
                ids = toks.input_ids[:, 1:].to(device) # remove cls token
                mask = toks.attention_mask[:, 1:].to(device)
                base_ids = model.tokenizer_base(seq,
                                add_special_tokens=True,
                                padding=False,
                                return_token_type_ids=False,
                                return_tensors='pt').input_ids.to(device)
                emb = model.embed(base_ids, ids, mask).float().detach().cpu().numpy()
                embeddings.append(emb)
        return embeddings
    return embed_double_model


def prepare_embed_protein_vec_dataset(model, aspect_token):
    def embed_protein_vec_dataset(seqs):
        with torch.no_grad():
            embeds = model.embed(seqs, aspect_token)
        return embeds.tolist()
    return embed_protein_vec_dataset


def embed_data(seqs,
               model,
               tokenizer,
               model_type='moesm',
               db_file='moesm_embeddings_clean.db'):
    sql = True
    device = torch.device('cuda')

    model.eval()
    embeddings = []
    batch_size = 1000

    if model_type == 'moesm':
        embed_seqs = prepare_embed_double_model(model, tokenizer, device)
    elif model_type == 'protvec':
        embed_seqs = prepare_embed_protein_vec_dataset(model, aspect_token='ALL')
    else:
        embed_seqs = prepare_embed_standard_model(model, tokenizer, device)

    for i in tqdm(range(0, len(seqs), batch_size), desc='Batches'):
        batch_seqs = seqs[i:i + batch_size]
        embs = embed_seqs(batch_seqs)
        if sql:
            with sqlite3.connect(db_file) as conn:
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
                for seq, emb in zip(batch_seqs, embs):
                    emb_data = np.array(emb).tobytes()
                    c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb_data))
                conn.commit()
        else:
            embeddings.extend(embs)
    if embeddings:
        return embeddings
    

if __name__ == '__main__':

    clean_datasets = glob('./CLEAN/data/*10.csv')
    model_type = 'ankh'
    db_file = 'ankh_base_10_clean.db'

    if model_type == None or db_file == None:
        model_type = input('Model type: ').lower()
        db_file = input('db path: ').lower()

    print(clean_datasets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'moesm':
        token = input('Token: ')
        model_path = 'lhallee/moesm_double_8_base'
        model = MoEsmVec.from_pretrained(model_path, token=token)
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    if model_type == 'esm':
        from transformers import EsmModel
        model_path = input('Esm path: ')
        model = EsmModel.from_pretrained(model_path)
        tokenizer = EsmTokenizer.from_pretrained(model_path)
    elif model_type == 'ankh':
        from transformers import T5EncoderModel, AutoTokenizer
        ankh_path = 'lhallee/ankh_base_encoder'
        model = T5EncoderModel.from_pretrained(ankh_path)
        tokenizer = AutoTokenizer.from_pretrained(ankh_path)
    elif model_type == 'protvec':
        from models.protein_vec.src_run.huggingface_protein_vec import ProteinVec, ProteinVecConfig
        from transformers import T5Tokenizer
        model_path = 'lhallee/ProteinVec'
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = ProteinVec.from_pretrained(model_path, config=ProteinVecConfig())
        model.to_eval()

    model = model.to(device)
    print(model)

    all_seqs = []
    for clean_dataset in clean_datasets:
        try:
            clean_seqs = set(pd.read_csv(clean_dataset, delimiter='\t')['Sequence'].tolist())
        except:
            clean_seqs = set(pd.read_csv(clean_dataset, delimiter='\t')['Sequences'].tolist())
        all_seqs.extend(clean_seqs)
    all_seqs = list(set(all_seqs))
    print(len(all_seqs), all_seqs[0])
    embed_data(all_seqs, model, tokenizer, model_type=model_type, db_file=db_file)
