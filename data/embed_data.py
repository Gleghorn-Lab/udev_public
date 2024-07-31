import torch
import sqlite3
from tqdm.auto import tqdm


def embed_dataset(cfg, model, sequences):
    full = cfg.full
    input_embeddings = []
    with torch.no_grad():
        for seq in tqdm(sequences, desc='Embedding'):  # Process embeddings in batches
            emb = model.embed(seq, full=full).detach().cpu().numpy()
            input_embeddings.append(emb)
    return input_embeddings


def embed_dataset_and_save(cfg, model, sequences):
    model.eval()
    db_file = cfg.db_path
    full = cfg.full
    batch_size = 1000
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc='Batches'):
                batch_sequences = sequences[i:i + batch_size]
                for seq in batch_sequences:  # Process embeddings in batches
                    emb = model.embed(seq, full=full).detach().cpu().numpy()
                    c.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, emb.tobytes()))
                conn.commit()
