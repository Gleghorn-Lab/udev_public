### imports
import torch
import random
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase
from typing import Any, Optional, Tuple, List, Dict
from transformers import EsmTokenizer


### PRETRAINING COLLATORS
class BasePretrainingCollator(DataCollatorMixin):
    """
    Base Data Collator Class to be inherited by
    more specialized collators

    Args: 

        tokenizer (EsmTokenizer): Tokenizes the input data.
        return_tensors (str, optional): The format of the returned tensors. Default is 'pt' (PyTorch).
        JEPA (bool, optional): Whether to use Joint Embedding Predictive Architecture. Default is False.
        **kwargs: Additional keyword arguments.

    Returns:
        batch: A dictionary containing 'input_ids', 'attention_mask', and 'labels' as keys, along with 'original_ids' if JEPA is enabled.
    """
    def __init__(self,
                 tokenizer,
                 return_tensors: str = 'pt',
                 JEPA: bool = False,
                 **kwargs):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.JEPA = JEPA

    def torch_call(self, inputs):
        batch, labels = self.torch_mask_tokens(inputs)
        batch['labels'] = labels
        if self.JEPA:
            batch['original_ids'] = batch['input_ids'].clone()
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        raise NotImplementedError
    
    """
    Masks tokens in the input data. This method should be implemented by subclasses.
    """


class DataCollatorForMLM(BasePretrainingCollator):
    """
    Data Collator for Masked Language Modeling tasks, inherits from BaseDataCollator
    This class prepares datasets for MLM by masking tokens in the input data
    according to a specified probability. 
    
    Args:
        tokenizer (EsmTokenizer): Tokenizes the input data.
        mlm_probability (float, optional): Probability with which tokens will be masked. Default is 0.15.
        bwd (bool, optional): Flag to enable showing the sequence backwards, half the time.
        When set to True, special tokens are not added (<cls> is omitted), as <eos> is added by SequenceDatasetBwd class
        **kwargs: Additional keyword arguments.
    Returns:
        batch: A dictionary containing 'input_ids', 'attention_mask', and 'labels' as keys, along with 'original_ids' if JEPA is enabled.
    """
    def __init__(self,
                 tokenizer, 
                 mlm_probability: float = 0.15,
                 max_length: int = 2048,
                 bwd: bool = False, **kwargs):
        
        super().__init__(tokenizer, **kwargs)
        self.mlm_probability = mlm_probability
        self.bwd = bwd
        self.max_length = max_length # TODO add to all

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        add_special_tokens = not self.bwd
        inputs = self.tokenizer(inputs,
                                return_tensors=self.return_tensors,
                                padding='longest',
                                max_length=self.max_length,
                                truncation=True,
                                add_special_tokens=add_special_tokens)
        labels = inputs.input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs.input_ids[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100
        return inputs, labels
    

class DataCollatorForSectionalMasking(BasePretrainingCollator):
    def __init__(self,
                 tokenizer,
                 noise_density: float = 0.15,
                 bwd: bool = False,
                 **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.noise_density = noise_density
        self.bwd = bwd

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        add_special_tokens = not self.bwd
        inputs = self.tokenizer(inputs, return_tensors=self.return_tensors, padding='longest', truncation=False, add_special_tokens=add_special_tokens)
        labels = inputs.input_ids.clone()
        labels.fill_(-100)

        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        for i, seq in enumerate(inputs.input_ids):
            seq_length = (seq != self.tokenizer.pad_token_id).sum()
            section_length = int(seq_length * self.noise_density)
            start_index = torch.randint(0, seq_length - section_length + 1, (1,)).item()
            indices_to_mask = torch.arange(start_index, start_index + section_length) % seq_length
            indices_to_mask = indices_to_mask[~special_tokens_mask[i, indices_to_mask]]

            labels[i, indices_to_mask] = inputs.input_ids[i, indices_to_mask]
            inputs.input_ids[i, indices_to_mask] = self.tokenizer.mask_token_id

        return inputs, labels
    


class DataCollatorForSectionalMasking(BasePretrainingCollator):
    def __init__(self,
                 tokenizer,
                 noise_density: float = 0.15,
                 bwd: bool = False,
                 **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.noise_density = noise_density
        self.bwd = bwd

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        add_special_tokens = not self.bwd
        inputs = self.tokenizer(inputs, return_tensors=self.return_tensors, padding='longest', truncation=False, add_special_tokens=add_special_tokens)
        labels = inputs.input_ids.clone()
        labels.fill_(-100)

        if special_tokens_mask is None:
            special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        for i, seq in enumerate(inputs.input_ids):
            seq_length = (seq != self.tokenizer.pad_token_id).sum()
            section_length = int(seq_length * self.noise_density)
            start_index = torch.randint(0, seq_length - section_length + 1, (1,)).item()
            indices_to_mask = torch.arange(start_index, start_index + section_length) % seq_length
            indices_to_mask = indices_to_mask[~special_tokens_mask[i, indices_to_mask]]

            labels[i, indices_to_mask] = inputs.input_ids[i, indices_to_mask]
            inputs.input_ids[i, indices_to_mask] = self.tokenizer.mask_token_id

        return inputs, labels
    

class DataCollatorForShuffling(BasePretrainingCollator):
    def __init__(self,
                 tokenizer,
                 noise_density: float = 0.15,
                 **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.noise_density = noise_density

    def shuffle_characters(self, seq):
        seq = list(seq)
        per = self.noise_density
        seq_len = len(seq)
        num_to_shuffle = int(seq_len * per)
        shuffled_indices = random.sample(range(seq_len), num_to_shuffle)
        shuffled_seq = seq[:]
        shuffled_idx = shuffled_indices[:]
        random.shuffle(shuffled_idx)
        for old_idx, new_idx in zip(shuffled_indices, shuffled_idx):
            shuffled_seq[new_idx] = f'<{seq[old_idx]}s>'
            
        return ''.join(shuffled_seq)
    
    def torch_mask_tokens(self, seqs):
        shuffled_seqs = [self.shuffle_characters(seq) for seq in seqs]
        original_labels = self.tokenizer(seqs,
                                         return_tensors=self.return_tensors,
                                         padding='longest',
                                         truncation=False,
                                         return_token_type_ids=False)
        labels = original_labels.input_ids.clone()
        inputs = self.tokenizer(shuffled_seqs,
                                return_tensors=self.return_tensors,
                                padding='longest',
                                truncation=False,
                                return_token_type_ids=False)
        apply_neg = labels.eq(inputs.input_ids)
        labels[apply_neg] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return inputs, labels


class DataCollatorForSectionalShuffling(BasePretrainingCollator):
    def __init__(self, tokenizer, noise_density: float = 0.15, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.noise_density = noise_density

    def shuffle_section(self, seq):
        seq = list(seq)
        seq_len = len(seq)
        per = self.noise_density
        section_length = int(seq_len * per)
        start_index = random.randrange(seq_len)
        indices = [(start_index + j) % len(seq) for j in range(section_length)]
        section = [seq[i] for i in indices]
        random.shuffle(section)
        section = [f'<{c}s>' for c in section]
        for i, sec in zip(indices, section):
            seq[i] = sec

        return ''.join(seq)

    def torch_mask_tokens(self, seqs):
        shuffled_seqs = [self.shuffle_section(seq) for seq in seqs]
        original_labels = self.tokenizer(seqs, return_tensors=self.return_tensors, padding='longest', truncation=False)
        labels = original_labels.input_ids.clone()
        inputs = self.tokenizer(shuffled_seqs, return_tensors=self.return_tensors, padding='longest', truncation=False, return_token_type_ids=False)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return inputs, labels


def get_pretraining_collator(tokenizer, mlm=False, shuffling=False, sectional=False, jepa=False, noise_density=0.15):
    if mlm:
        if sectional:
            return DataCollatorForSectionalMasking(tokenizer=tokenizer, noise_density=noise_density, JEPA=jepa)
        else:
            return DataCollatorForMLM(tokenizer=tokenizer, mlm_probability=noise_density, JEPA=jepa)
    elif shuffling:
        if sectional:
            return DataCollatorForSectionalShuffling(tokenizer=tokenizer, noise_density=noise_density, JEPA=jepa)
        else:
            return DataCollatorForShuffling(tokenizer=tokenizer, noise_density=noise_density, JEPA=jepa)
    else:
        raise ValueError("At least one of mlm, shuffling must be True.")


class AutoEncoderCollator(DataCollatorMixin):
    def __init__(self,
                 tokenizer,
                 return_tensors: str = 'pt',
                 max_length: int = 2048,
                 **kwargs):
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.max_length = max_length

    def torch_call(self, inputs):
        batch = self.tokenizer(inputs,
                                return_tensors=self.return_tensors,
                                padding='longest',
                                max_length=self.max_length,
                                truncation=True,
                                add_special_tokens=True)
        labels = batch['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch['labels'] = labels
        return batch


class DataCollatorForMTL(DataCollatorMixin):
    """
    Custom collate function for handling variable-sized t_labels and tokenization after batching (multitask learning).
    """
    
    mlm: bool = False
    min_prob: float = 0.05
    max_prob: float = 0.15
    return_tensors: str = 'pt'
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    domains: Optional[List] = None
    assert min_prob <= max_prob

    def torch_call(self, input):
        if isinstance(input[0], tuple):
            seqs_batch = [item[0] for item in input]
            t_labels_batch = [item[1] for item in input] # t_labels are different sizes, so place in list
            r_labels_batch = torch.stack([item[2] for item in input])
        else: # if batch_size 1
            seqs_batch = [input[0]]
            t_labels_batch = [input[1]]
            r_labels_batch = torch.tensor(input[2])

        # Tokenize the sequences after batching
        tokenized = self.tokenizer(seqs_batch, return_tensors='pt', padding='longest')

        if self.domains != None:
            for i, r_label in enumerate(r_labels_batch):
                domain_token = self.tokenizer(self.domains[int(r_label.item())], add_special_tokens=False).input_ids[0]
                tokenized['input_ids'][i][0] = domain_token

        batch = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'router_labels': r_labels_batch,
            'labels': t_labels_batch
        }

        if self.mlm:
            batch['input_ids'], mask_labels = self.torch_mask_tokens(batch['input_ids'])
            if self.tokenizer.pad_token_id is not None:
                mask_labels[mask_labels == self.tokenizer.pad_token_id] = -100
            batch['mask_labels'] = mask_labels

        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 90% MASK, 5% random, 5% original.
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, random.uniform(self.min_prob, self.max_prob))
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        # 85% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.90)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced # 0.5 because half of remaining are random
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (5% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def pair_collator(inputs, tokenizer):
    seqa = [f[0] for f in inputs]
    seqb = [f[1] for f in inputs]
    labels = [f[2] for f in inputs]
    a = tokenizer(seqa, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=True)
    b = tokenizer(seqb, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=True)
    max_batch_length = len(max(labels, key=len))
    labels = torch.stack([torch.tensor(label + [-100] * (max_batch_length - len(label))) for label in labels])
    return {
        'seq_a': a,
        'seq_b': b,
        'labels': labels
    }


def collate_seq_labels(tokenizer):
    def _collate_fn(batch):
        seqs = [ex[0] for ex in batch]
        labels = [ex[1] for ex in batch]
        batch = tokenizer(seqs,
                          padding='longest',
                          truncation=False,
                          return_tensors='pt',
                          add_special_tokens=True)
        batch['labels'] = torch.stack([torch.tensor(label, dtype=torch.float) for label in labels])
        return batch
    return _collate_fn


def collate_fn_embeds(full=False, max_length=512, task_type='tokenwise'):
    def _collate_fn(batch):
        embeds = torch.stack([ex['embeddings'] for ex in batch])
        if full and task_type == 'tokenwise':
            labels = [ex['labels'] for ex in batch]
            padded_labels = []         
            for label in labels:
                padding_size = max_length - label.size(0)
                if padding_size > 0:
                    padding = torch.full((padding_size,), -100, dtype=label.dtype)
                    padded_label = torch.cat((label.squeeze(-1), padding))
                else:
                    padded_label = label[:max_length].squeeze(-1)  # Truncate if longer than max_length
                padded_labels.append(padded_label)
            labels = torch.stack(padded_labels)
        else:
            labels = torch.stack([ex['labels'] for ex in batch])
        return {
            'embeddings': embeds,
            'labels': labels
        }
    return _collate_fn


def standard_data_collator(batch):
    batch = {k: torch.stack([ex[k] for ex in batch]) for k in batch[0].keys()}


def vision_collator(batch):
    imgs = torch.stack([item['img'] for item in batch])
    if 'labels' in batch[0]:
        if isinstance(batch[0]['labels'], torch.Tensor):
            labels = torch.stack([item['labels'] for item in batch])
        else:
            labels = torch.tensor([item['labels'] for item in batch])
        return {'img': imgs, 'labels': labels}
    else:
        return {'img': imgs}


class CampCollator(DataCollatorForMLM):
    def __init__(self,
                 plm_tokenizer,
                 max_length_plm=2048,
                 max_length_nlp=1024,
                 nlp_tokenizer=None,
                 num_annotations=32000,
                 annotation_transformer=True,
                 mlm_probability=0.15,
                 masking=False,
                 **kwargs):
        assert annotation_transformer or nlp_tokenizer, 'Needs annotation transformer OR nlp_tokenizer'
        self.return_tensors = 'pt'
        self.tokenizer = plm_tokenizer
        self.nlp_tokenizer = nlp_tokenizer
        self.annotation_transformer = annotation_transformer
        self.vocab_size = num_annotations
        self.pad_token_id = 0
        self.cls_id = self.vocab_size - 3
        self.eos_id = self.vocab_size - 2
        self.max_length = max_length_plm # goes to MLM collator
        self.max_length_nlp = max_length_nlp
        self.mlm_probability = mlm_probability
        self.masking = masking

        self.bwd = False # to appease DataCollatorForMLM

    def _pad_sequences(self, sequences, padding_value):
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [padding_value] * (max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)

    def _create_attention_mask(self, input_ids, padding_value):
        attention_mask =  (input_ids != padding_value).long()
        return attention_mask
    
    def collate_annotations(self, anns):
        input_ids = [[self.cls_id] + ann + [self.eos_id] for ann in anns]
        input_ids = self._pad_sequences(input_ids, self.pad_token_id)
        batch = {
            'input_ids': input_ids,
            'attention_mask': self._create_attention_mask(input_ids, self.pad_token_id),
        }
        return batch

    def torch_call(self, inputs):
        proteins = [f[0] for f in inputs]
        descriptions = [f[1] for f in inputs]

        if self.annotation_transformer:
            nlp_tok = self.collate_annotations(descriptions)
        else:
            nlp_tok = self.nlp_tokenizer(
                descriptions,
                padding='longest',
                truncation=True,
                max_length=self.max_length_nlp,
                return_tensors='pt'
            )

        if self.masking:
            plm_tok, labels = self.torch_mask_tokens(proteins)
            plm_tok['labels'] = labels
            plm_tok['original_ids'] = plm_tok['input_ids'].clone()
        else:
            plm_tok = self.tokenizer(proteins,
                        return_tensors=self.return_tensors,
                        padding='longest',
                        max_length=self.max_length,
                        truncation=True,
                        add_special_tokens=True)
        return {
            'plm_tok':plm_tok,
            'nlp_tok':nlp_tok,
            'labels':torch.ones((len(proteins),)) # for similarity metrics
        }


class AnnotationCollator:
    def __init__(self, vocab_size, mlm_probability: float = 0.15):
        self.mlm_probability = mlm_probability
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_id = vocab_size + 1
        self.eos_id = vocab_size + 2
        self.mask_token_id = vocab_size + 3
        self.special_token_ids = [0, self.cls_id, self.eos_id, self.mask_token_id]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [[self.cls_id] + example['input_ids'] + [self.eos_id] for example in examples]
        input_ids = self._pad_sequences(input_ids)
        batch = {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': self._create_attention_mask(input_ids),
        }
        batch['input_ids'], batch['labels'] = self._mask_tokens(batch['input_ids'], batch['labels'])
        return batch

    def _pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)

    def _create_attention_mask(self, input_ids: List[List[int]]) -> torch.Tensor:
        attention_mask =  (input_ids != self.pad_token_id).long()
        return attention_mask

    def _mask_tokens(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_tokens_mask |= (inputs == token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels


class SequenceAnnotationCollator:
    def __init__(self, plm_tokenizer, mlm_probability: float = 0.15,
                 both_weight: float = 0.5, seq_weight: float = 0.25, ann_weight: float = 0.25,
                 mask_sequence: bool = True, mask_annotation: bool = True):
        self.tokenizer = plm_tokenizer
        self.mlm_probability = mlm_probability
        self.plm_vocab = plm_tokenizer.vocab_size
        self.pad_token_id = plm_tokenizer.pad_token_id
        self.cls_id = plm_tokenizer.cls_token_id
        self.eos_id = plm_tokenizer.eos_token_id
        self.mask_token_id = plm_tokenizer.mask_token_id
        self.special_token_ids = [self.pad_token_id, self.cls_id, self.eos_id, self.mask_token_id]
        
        # Strategy weights
        self.strategy_weights = [both_weight, seq_weight, ann_weight]
        if sum(self.strategy_weights) != 1.0:
            raise ValueError("The sum of strategy weights must be 1.0")
        if both_weight == 0.0 and seq_weight == 0.0 and mask_sequence:
            raise ValueError(f"mask_sequence {mask_sequence} and both_weight {both_weight} and seq_weight {seq_weight} cannot both be 0.0")
        if both_weight == 0.0 and ann_weight == 0.0 and mask_annotation:
            raise ValueError(f"mask_annotation {mask_annotation} and both_weight {both_weight} and ann_weight {ann_weight} cannot both be 0.0")

        # Masking options
        self.mask_sequence = mask_sequence
        self.mask_annotation = mask_annotation

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [example[0] for example in examples]
        anns = [example[1] for example in examples]

        # Tokenize sequences
        tokenized_seqs = self.tokenizer(seqs, padding=False, truncation=False, add_special_tokens=False)

        # Decide on the strategy for each example
        strategies = random.choices(['both', 'seq', 'ann'], weights=self.strategy_weights, k=len(examples))

        # Combine sequence and annotation IDs based on the strategy
        input_ids = []
        seq_lengths = []  # To keep track of sequence lengths for masking
        for seq_ids, ann, strategy in zip(tokenized_seqs['input_ids'], anns, strategies):
            if strategy == 'both':
                ids = [self.cls_id] + seq_ids + [self.eos_id] + [ann_id + self.plm_vocab for ann_id in ann] + [self.eos_id]
                seq_lengths.append(len(seq_ids) + 2)  # +2 for CLS and EOS tokens
            elif strategy == 'seq':
                ids = [self.cls_id] + seq_ids + [self.eos_id]
                seq_lengths.append(len(ids))
            else:  # strategy == 'ann'
                ids = [self.cls_id] + [ann_id + self.plm_vocab for ann_id in ann] + [self.eos_id]
                seq_lengths.append(0)  # No sequence to mask
            input_ids.append(ids)

        input_ids = self._pad_sequences(input_ids)
        attention_mask = self._create_attention_mask(input_ids)

        batch = {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': attention_mask,
        }

        batch['input_ids'], batch['labels'] = self._mask_tokens(batch['input_ids'], batch['labels'], seq_lengths)
        return batch
    
    def _pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)

    def _create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_token_id).long()
        return attention_mask

    def _mask_tokens(self, inputs: torch.Tensor, labels: torch.Tensor, seq_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_tokens_mask |= (inputs == token_id)
        
        # Create masks for sequences and annotations
        seq_mask = torch.zeros_like(inputs, dtype=torch.bool)
        ann_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            seq_mask[i, :seq_len] = True
            ann_mask[i, seq_len:] = True
        
        # Apply masking options
        if not self.mask_sequence:
            probability_matrix.masked_fill_(seq_mask, value=0.0)
        if not self.mask_annotation:
            probability_matrix.masked_fill_(ann_mask, value=0.0)
        
        # Set probability of masking special tokens to 0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Ensure at least one non-special token is masked for each sequence if possible
        non_special_mask = ~special_tokens_mask & ((seq_mask & self.mask_sequence) | (ann_mask & self.mask_annotation))
        for i in range(inputs.size(0)):
            if non_special_mask[i].any():
                rand_index = torch.randint(0, non_special_mask[i].sum(), (1,))
                probability_matrix[i, non_special_mask[i].nonzero(as_tuple=True)[0][rand_index]] = 1.0
        
        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # Set labels for non-masked tokens to -100 (ignore in loss computation)
        labels[~masked_indices] = -100
        # Replace masked input tokens with mask token
        inputs[masked_indices] = self.mask_token_id
        return inputs, labels


class AspectEvaluationCollator(SequenceAnnotationCollator):
    def __init__(self, aspect_ranges, aspect, tokenizer, both=False, AT=False):
        self.start, self.end = aspect_ranges[aspect]
        max = 0
        for _, v in aspect_ranges.items():
            for m in v:
                if m > max:
                    max = m
        vocab_size = max

        self.both = both
        self.AT = AT
        self.tokenizer = tokenizer
        self.plm_vocab = tokenizer.vocab_size

        if AT:
            self.pad_token_id = 0
            self.cls_id = vocab_size + 1
            self.eos_id = vocab_size + 2
            self.mask_token_id = vocab_size + 3
        else:
            self.pad_token_id = tokenizer.pad_token_id
            self.cls_id = tokenizer.cls_token_id
            self.eos_id = tokenizer.eos_token_id
            self.mask_token_id = tokenizer.mask_token_id
            self.start, self.end = self.start + self.plm_vocab, self.end + self.plm_vocab
        self.special_token_ids = [self.pad_token_id, self.cls_id, self.eos_id, self.mask_token_id]

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [example['input_ids'] for example in examples]
        anns = [example['attention_mask'] for example in examples]

        if not self.AT:
            anns = [[x + self.plm_vocab for x in sublist] for sublist in anns]

        # Tokenize sequences
        tokenized_seqs = self.tokenizer(seqs, padding=False, truncation=False, add_special_tokens=False)

        # Decide on the strategy for each example
        strategies = ['both'] * len(examples) if self.both else ['ann'] * len(examples)

        # Combine sequence and annotation IDs based on the strategy
        input_ids = []
        for seq_ids, ann, strategy in zip(tokenized_seqs['input_ids'], anns, strategies):
            if strategy == 'both':
                ids = [self.cls_id] + seq_ids + [self.eos_id] + ann + [self.eos_id]
            else:  # strategy == 'ann'
                ids = [self.cls_id] + ann + [self.eos_id]
            input_ids.append(ids)

        input_ids = self._pad_sequences(input_ids)
        attention_mask = self._create_attention_mask(input_ids)

        batch = {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'attention_mask': attention_mask,
        }

        batch['input_ids'], batch['labels'] = self._mask_tokens(batch['input_ids'], batch['labels'])
        return batch

    def _mask_tokens(self, inputs: torch.Tensor, labels: torch.Tensor):
        probability_matrix = torch.zeros(inputs.shape, dtype=torch.float32)
        aspect_mask = (inputs >= self.start) & (inputs <= self.end)
        probability_matrix[aspect_mask] = 1.0  # Always mask this aspect
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_tokens_mask |= (inputs == token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = probability_matrix.bool()        
        labels[~masked_indices] = -100
        inputs[masked_indices] = self.mask_token_id
        return inputs, labels

