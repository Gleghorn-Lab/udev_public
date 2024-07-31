import numpy as np
from datasets import load_dataset
from .data_utils import get_seqs, label_type_checker, encode_labels, process_seq_columns


def load_data_for_probe(args):
    print('\n-----Load Data-----\n')
    all_seqs, train_sets, valid_sets, test_sets, num_labels, task_types = [], [], [], [], [], []
    for data_path in args.data_paths:
        train_set, valid_set, test_set, num_label, task_type = get_data(args, data_path)
        num_labels.append(num_label)
        task_types.append(task_type)

        # Handle both single-sequence and PPI datasets
        train_data = get_seqs(train_set)
        valid_data = get_seqs(valid_set)
        test_data = get_seqs(test_set)

        if len(train_data) == 2:  # Single-sequence dataset
            train_seqs, train_labels = train_data
            valid_seqs, valid_labels = valid_data
            test_seqs, test_labels = test_data
            all_seqs.extend(train_seqs + valid_seqs + test_seqs)
        else:  # PPI dataset
            train_a_seqs, train_b_seqs, train_labels = train_data
            valid_a_seqs, valid_b_seqs, valid_labels = valid_data
            test_a_seqs, test_b_seqs, test_labels = test_data
            all_seqs.extend(train_a_seqs + train_b_seqs + valid_a_seqs + valid_b_seqs + test_a_seqs + test_b_seqs)
            train_seqs = (train_a_seqs, train_b_seqs)
            valid_seqs = (valid_a_seqs, valid_b_seqs)
            test_seqs = (test_a_seqs, test_b_seqs)

        train_sets.append((train_seqs, train_labels))
        valid_sets.append((valid_seqs, valid_labels))
        test_sets.append((test_seqs, test_labels))

    all_seqs = list(set(all_seqs))
    return all_seqs, train_sets, valid_sets, test_sets, num_labels, task_types


def get_data(cfg, data_path):
    label_col = 'labels'
    dataset = load_dataset(data_path)
    train_set, valid_set, test_set = dataset['train'], dataset['valid'], dataset['test']
    if not cfg.HF:
        train_set = process_seq_columns(train_set)
        valid_set = process_seq_columns(valid_set)
        test_set = process_seq_columns(test_set)
    if cfg.trim:
        original_train_size, original_valid_size, original_test_size = len(train_set), len(valid_set), len(test_set)
        
        if cfg.ppi:
            train_set = train_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= cfg.max_length)
            valid_set = valid_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= cfg.max_length)
            test_set = test_set.filter(lambda x: len(x['SeqA']) + len(x['SeqB']) <= cfg.max_length)
        else:
            train_set = train_set.filter(lambda x: len(x['seqs']) <= cfg.max_length)
            valid_set = valid_set.filter(lambda x: len(x['seqs']) <= cfg.max_length)
            test_set = test_set.filter(lambda x: len(x['seqs']) <= cfg.max_length)
        
        print(f'Trimmed {round((original_train_size-len(train_set))/original_train_size, 2)}% from train')
        print(f'Trimmed {round((original_valid_size-len(valid_set))/original_valid_size, 2)}% from valid')
        print(f'Trimmed {round((original_test_size-len(test_set))/original_test_size, 2)}% from test')
    
    check_labels = valid_set[label_col]
    label_type = label_type_checker(check_labels)

    if label_type == 'string':
        example = valid_set[label_col][0]
        try:
            import ast
            new_ex = ast.literal_eval(example)
            if isinstance(new_ex, list): # if ast runs correctly and is now a list it is multilabel labels
                label_type = 'multilabel'
                train_set = train_set.map(lambda ex: {label_col: ast.literal_eval(ex[label_col])})
                valid_set = valid_set.map(lambda ex: {label_col: ast.literal_eval(ex[label_col])})
                test_set = test_set.map(lambda ex: {label_col: ast.literal_eval(ex[label_col])})
        except:
            label_type = 'string' # if ast throws error it is actually string
            id_dicts = []

    if label_type == 'string':
        train_labels = train_set[label_col]
        unique_tags = set(tag for doc in train_labels for tag in doc)
        tag2id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
        id2tag = {id: tag for tag, id in tag2id.items()}
        id_dicts.append(id2tag)
        train_set = train_set.map(lambda ex: {label_col: encode_labels(ex[label_col], tag2id=tag2id)})
        valid_set = valid_set.map(lambda ex: {label_col: encode_labels(ex[label_col], tag2id=tag2id)})
        test_set = test_set.map(lambda ex: {label_col: encode_labels(ex[label_col], tag2id=tag2id)})
        label_type = 'tokenwise'
        num_labels = len(unique_tags)
    else:
        if label_type == 'regression':
            num_labels = 1
        else:
            try:
                num_labels = len(train_set[label_col][0])
            except:
                num_labels = len(np.unique(train_set[label_col]))

    return train_set, valid_set, test_set, num_labels, label_type



### tests
if __name__ == 'main':
    pass
