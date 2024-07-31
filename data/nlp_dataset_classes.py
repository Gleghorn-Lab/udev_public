### imports
import random
import torch
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm
from typing import List, Dict, Tuple


class PairDataset(TorchDataset):
    def __init__(self, data, col_a, col_b, label_col):
        self.seqs_a = data[col_a]
        self.seqs_b = data[col_b]
        self.labels = data[label_col]

    def avg(self):
        return sum(len(seqa) + len(seqb) for seqa, seqb in zip(self.seqs_a, self.seqs_b)) / len(self.seqs_a)

    def __len__(self):
        return len(self.seqs_a)

    def __getitem__(self, idx):
        return self.seqs_a[idx], self.seqs_b[idx], self.labels[idx]


class SequenceDataset(TorchDataset):    
    def __init__(self, dataset, col_name='seqs'):
        self.seqs = dataset[col_name]
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq


class SequenceLabelDataset(TorchDataset):    
    def __init__(self, dataset, col_name='seqs', label_col='labels'):
        self.seqs = dataset[col_name]
        self.labels = dataset[label_col]
        self.lengths = [len(seq) for seq in self.seqs]

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label


class SequenceDatasetBwd(TorchDataset):
    def __init__(self, dataset, col_name='seqs', eos_token='<eos>', fwd_token='<fwd>', bwd_token='<bwd>'):
        self.seqs = dataset[col_name]
        self.lengths = [len(seq) for seq in self.seqs]
        self.eos_token = eos_token
        self.fwd_token = fwd_token
        self.bwd_token = bwd_token

    def avg(self):
        return sum(self.lengths) / len(self.lengths)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if random.random() > 0.5:
            seq = seq[::-1]
            seq = self.bwd_token + seq + self.eos_token
        else:
            seq = self.fwd_token + seq + self.eos_token
        return seq


class FineTuneDatasetEmbedsFromDisk(TorchDataset):
    def __init__(self, cfg, seqs, labels, input_dim=768, task_type='binary', all_seqs = None): 
        self.db_file = cfg.db_path
        self.batch_size = cfg.batch_size
        self.emb_dim = input_dim
        self.full = cfg.full
        self.seqs, self.labels = seqs, labels
        self.length = len(labels)
        self.max_length = len(max(seqs, key=len))
        print('Max length: ', self.max_length)
        self.task_type = task_type
        self.read_amt = cfg.read_scaler * self.batch_size
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

        if all_seqs:
            print('Pre shuffle check')
            self.check_seqs(all_seqs)
        self.reset_epoch()
        if all_seqs:
            print('Post shuffle check')
            self.check_seqs(all_seqs)

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        cond = False
        for seq in self.seqs:
            if seq not in all_seqs:
                cond = True
            if cond:
                break
        if cond:
            print('Sequences not found in embeddings')
        else:
            print('All sequences in embeddings')


    def reset_epoch(self):
        data = list(zip(self.seqs, self.labels))
        random.shuffle(data)
        self.seqs, self.labels = zip(*data)
        self.seqs, self.labels = list(self.seqs), list(self.labels)
        self.embeddings, self.current_labels = [], []
        self.count, self.index = 0, 0

    def read_embeddings(self):
        embeddings, labels = [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (self.seqs[i],))
            row = result.fetchone()
            emb_data = row[0]
            emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.emb_dim))
            if self.full:
                padding_needed = self.max_length - emb.size(0)
                emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
            embeddings.append(emb)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings = embeddings
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb = self.embeddings[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)

        return {'embeddings': emb.squeeze(0), 'labels': label}


class FineTuneDatasetEmbeds(TorchDataset):
    def __init__(self, cfg, emb_dict, seqs, labels, task_type='binary'):
        self.embeddings = self.get_embs(emb_dict, seqs)
        self.labels = labels
        self.task_type = task_type
        self.max_length = len(max(seqs, key=len))
        print('Max length: ', self.max_length)
        self.full = cfg.full

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict.get(seq)
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        if self.task_type == 'multilabel' or self.task_type == 'regression':
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        emb = torch.tensor(self.embeddings[idx])
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
        return {'embeddings': emb.squeeze(0), 'labels': label}


class PPIDatasetEmbedsFromDisk(TorchDataset):
    def __init__(self, cfg, seqs_a, seqs_b, labels, input_dim=768, all_seqs=None):
        self.db_file = cfg.db_path
        self.batch_size = cfg.batch_size
        self.emb_dim = input_dim
        self.full = cfg.full
        self.seqs_a, self.seqs_b, self.labels = seqs_a, seqs_b, labels
        self.length = len(labels)
        lengths = [len(a) + len(b) for a, b in zip(seqs_a, seqs_b)]
        self.max_length = max(lengths)
        print('Max length: ', self.max_length)
        self.read_amt = cfg.read_scaler * self.batch_size
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0

        if all_seqs:
            print('Pre shuffle check')
            self.check_seqs(all_seqs)
        self.reset_epoch()
        if all_seqs:
            print('Post shuffle check')
            self.check_seqs(all_seqs)

    def __len__(self):
        return self.length

    def check_seqs(self, all_seqs):
        cond = False
        for a, b in zip(self.seqs_a, self.seqs_b):
            if a not in all_seqs or b not in all_seqs:
                cond = True
            if cond:
                break
        if cond:
            print('Sequences not found in embeddings')
        else:
            print('All sequences in embeddings')

    def reset_epoch(self):
        data = list(zip(self.seqs_a, self.seqs_b, self.labels))
        random.shuffle(data)
        self.seqs_a, self.seqs_b, self.labels = zip(*data)
        self.seqs_a, self.seqs_b, self.labels = list(self.seqs_a), list(self.seqs_b), list(self.labels)
        self.embeddings_a, self.embeddings_b, self.current_labels = [], [], []
        self.count, self.index = 0, 0

    def get_embedding(self, c, seq):
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        emb_data = row[0]
        emb = torch.tensor(np.frombuffer(emb_data, dtype=np.float32).reshape(-1, self.emb_dim))
        if self.full:
            padding_needed = self.max_length - emb.size(0)
            emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
        return emb

    def read_embeddings(self):
        embeddings_a, embeddings_b, labels = [], [], []
        self.count += self.read_amt
        if self.count >= self.length:
            self.reset_epoch()
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        for i in range(self.count, self.count + self.read_amt):
            if i >= self.length:
                break
            emb_a = self.get_embedding(c, self.seqs_a[i])
            emb_b = self.get_embedding(c, self.seqs_b[i])
            embeddings_a.append(emb_a)
            embeddings_b.append(emb_b)
            labels.append(self.labels[i])
        conn.close()
        self.index = 0
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b
        self.current_labels = labels

    def __getitem__(self, idx):
        if self.index >= len(self.current_labels) or len(self.current_labels) == 0:
            self.read_embeddings()

        emb_a = self.embeddings_a[self.index]
        emb_b = self.embeddings_b[self.index]
        label = self.current_labels[self.index]

        self.index += 1

        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a

        # Stack the embeddings
        emb = torch.cat([emb_a, emb_b], dim=1)

        label = torch.tensor(label, dtype=torch.long)

        return {'embeddings': emb.squeeze(0), 'labels': label}


class PPIDatasetEmbeds(TorchDataset):
    def __init__(self, cfg, emb_dict, seqs_a, seqs_b, labels):
        self.embeddings_a = self.get_embs(emb_dict, seqs_a)
        self.embeddings_b = self.get_embs(emb_dict, seqs_b)
        self.labels = labels
        self.max_length = max(len(max(seqs_a, key=len)), len(max(seqs_b, key=len)))
        print('Max length: ', self.max_length)
        self.full = cfg.full

    def __len__(self):
        return len(self.labels)
    
    def get_embs(self, emb_dict, seqs):
        embeddings = []
        for seq in tqdm(seqs, desc='Loading Embeddings'):
            emb = emb_dict.get(seq)
            embeddings.append(emb)
        return embeddings

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        emb_a = torch.tensor(self.embeddings_a[idx])
        emb_b = torch.tensor(self.embeddings_b[idx])
        
        # 50% chance to switch the order of a and b
        if random.random() < 0.5:
            emb_a, emb_b = emb_b, emb_a
        
        # Stack the embeddings
        emb = torch.cat([emb_a, emb_b], dim=1)
        
        if self.full:
            padding_needed = 2 * self.max_length - emb.size(0)
            emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_needed), value=0)
        
        return {'embeddings': emb.squeeze(0), 'labels': label}


class CampDataset(TorchDataset):
    def __init__(self, data, seq_col, ann_col):
        self.anns = data[ann_col]
        self.seqs = data[seq_col]

    def avg(self):
        return sum(len(seq) for seq in self.seqs) / len(self.seqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.anns[idx]


class CodonTranslationDataset(TorchDataset):
    def __init__(self, data, aa_col, codon_col, tag_col):
        self.aas = data[aa_col]
        self.codons = data[codon_col]
        self.tags = data[tag_col]

    def avg(self):
        return sum(len(seq) for seq in self.aas) / len(self.seqs)

    def __len__(self):
        return len(self.aas)

    def __getitem__(self, idx):
        return self.aas[idx], self.codons[idx], self.tags[idx]


class TokenizedDataset(TorchDataset):
    def __init__(self, dataset, col_name):
        self.inputs = dataset[col_name]
        self.inputs = [sorted(input) for input in self.inputs]

    def avg(self):
        return sum(len(input) for input in self.inputs) / len(self.inputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx]}


class SequenceAnnotationDataset(TorchDataset):
    def __init__(self, dataset, seq_col, ann_col, max_length: int = 512):
        self.seqs = dataset[seq_col]
        self.anns = dataset[ann_col]
        self.max_length = max_length

    def avg(self):
        total_len = sum(self.get_total_length(self.seqs[i], self.anns[i]) for i in range(len(self)))
        return total_len / len(self)

    def shuffle(self):
        data = list(zip(self.seqs, self.anns))
        random.shuffle(data)
        self.seqs, self.anns = zip(*data)
        self.seqs, self.anns = list(self.seqs), list(self.anns)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.process_sequence_and_annotations(self.seqs[idx], self.anns[idx]) # seq, ann

    def get_total_length(self, sequence: str, annotations: List[int]) -> int:
        return len(sequence) + len(annotations)

    def process_sequence_and_annotations(self, sequence: str, annotations: List[int]) -> Tuple[str, List[int]]:
        total_length = self.get_total_length(sequence, annotations)
        if total_length <= self.max_length:
            return sequence, sorted(annotations)
        ann_length, seq_length = len(annotations), len(sequence)
        extra = total_length - self.max_length
        max_ann_reduce = int(ann_length * (3/4)) # keep at a minimum a quarter of the annotations
        if extra > max_ann_reduce:
            new_ann_length = ann_length - max_ann_reduce
            seq_reduce = extra - max_ann_reduce
        else:
            new_ann_length = ann_length - extra
            seq_reduce = 0
        shuffled_anns = annotations.copy()
        random.shuffle(shuffled_anns)
        processed_anns = sorted(shuffled_anns[:new_ann_length])
        processed_seq = sequence[:seq_length - seq_reduce]
        return processed_seq, processed_anns

    def hist(self, bins=50, figsize=(12, 6), save_path=None, call=False):
        if call:
            lengths = [self.get_total_length(*self.process_sequence_and_annotations(self.seqs[i], self.anns[i]))
                       for i in range(len(self))]
        else:
            lengths = [self.get_total_length(self.seqs[i], self.anns[i]) for i in range(len(self))]
        plt.figure(figsize=figsize)
        plt.hist(lengths, bins=bins, edgecolor='black')
        plt.title('Distribution of Combined Sequence and Annotation Lengths')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        if self.max_length:
            plt.axvline(x=self.max_length, color='r', linestyle='dashed', linewidth=2)
            plt.text(self.max_length, plt.ylim()[1], f'Max Length: {self.max_length}', 
                     rotation=90, va='top', ha='right', color='r')
        if save_path:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        else:
            plt.show()
        plt.close()


### tests
if __name__ == 'main':
    pass

