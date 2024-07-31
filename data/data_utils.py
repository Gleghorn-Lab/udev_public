import re
import ast
import torch
import sqlite3
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as VF


def build_pair_filter_function(col_a, col_b, max_len, min_len):
    """
    Returns function for trimming HF datasets with two columns by their length
    inputs
    col_a, col_b - string column names
    min_len, max_len - minimum length and maximum combined length (int)
    returns
    function for use with datasets.Dataset.map()
    """
    def filter_func(x):
        len_a, len_b = len(x[col_a]), len(x[col_b])
        cond1 = len_a + len_b <= max_len
        cond2 = len_a >= min_len and len_b >= min_len
        return cond1 and cond2
    return filter_func


def get_max_from_list_of_lists(lst):
    """
    Given a nested list, return the maximum value in all possible elements
    """
    return max(max(sub_list) for sub_list in lst)


def process_column(example, col_name):
    """
    For use with datasets.Dataset.map(), returns the ast literal (string to list or dict, etc.)
    """
    example[col_name] = ast.literal_eval(example[col_name])
    return example


def not_regression(labels): # not a great assumption but works most of the time
    return all(isinstance(label, (int, float)) and label == int(label) for label in labels)


def encode_labels(labels, tag2id):
    return [torch.tensor([tag2id[tag] for tag in doc], dtype=torch.long) for doc in labels]


def label_type_checker(labels):
    ex = labels[0]
    if not_regression(labels):
        if isinstance(ex, list):
            label_type = 'multilabel'
        elif isinstance(ex, int) or isinstance(ex, float):
            label_type = 'singlelabel' # binary or multiclass
    elif isinstance(ex, str):
        label_type = 'string'
    else:
        label_type = 'regression'
    return label_type


def process_seq_columns(dataset):
    for column in dataset.column_names:
        if 'seq' in column.lower():
            dataset = dataset.map(
                lambda example: {
                    column: re.sub(r"[UZOB]", "X", example[column]).replace(' ', '')
                }
            )
    return dataset


def get_seqs(dataset, seq_col='seqs', label_col='labels', seq_a_col='SeqA', seq_b_col='SeqB'):
    if dataset.num_columns > 2:  # Assuming it's a PPI dataset if there are more than 2 columns
        if seq_a_col in dataset.column_names and seq_b_col in dataset.column_names:
            return dataset[seq_a_col], dataset[seq_b_col], dataset[label_col]
        else:
            raise ValueError(f"Columns {seq_a_col} and {seq_b_col} not found in the dataset. Please specify correct column names for PPI sequences.")
    else:
        return dataset[seq_col], dataset[label_col]


def read_sequences_from_db(db_file):
    sequences = []
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        c.execute("SELECT sequence FROM embeddings")
        rows = c.fetchall()
        for row in rows:
            sequences.append(row[0])
    return sequences


def calculate_mean_std(imgs):
    num_channels = np.array(imgs)[0].shape[2]  # Get the number of channels from the first image
    data = [np.dstack([np.array(img, dtype=np.float32)[:, :, c] / 255.0 for img in imgs]) for c in range(num_channels)]
    mean = tuple(np.mean(channel_data) for channel_data in data)
    std = tuple(np.std(channel_data) for channel_data in data)
    return mean, std


def preview_crops(imgs, GTs):
    imgs = imgs.permute(0, 2, 3, 1)
    GTs = GTs.permute(0, 2, 3, 1)
    for i in range(len(imgs)):
        fig = plt.figure(figsize=(5, 4))
        fig.add_subplot(1, 2, 1)
        plt.imshow(imgs[i], cmap='viridis')
        plt.axis('off')
        plt.title('Img')
        fig.add_subplot(1, 2, 2)
        plt.imshow(GTs[i][:, :, 0], cmap='gray')
        plt.axis('off')
        plt.title('GT')
        plt.show()


def viewer(pred, gt, val=False):
    print(pred.shape, gt.shape)
    plot = np.hstack((pred, gt))
    if val:
        ratio = 0.5
        plot = np.array(Image.fromarray(plot).resize((int(plot.shape[1] * ratio), int(plot.shape[0] * ratio))))
    plt.imshow(plot)
    plt.show()


def color_jitter(img):
    jitter_bri = ColorJitter(brightness=(0.05, 0.2))
    jitter_hue = ColorJitter(hue=(-0.05, 0.05))
    jitter_con = ColorJitter(contrast=(0.05, 0.2))
    jitter_sat = ColorJitter(saturation=(0.05, 0.2))
    choice = random.choice([1, 2, 3, 4])
    if choice == 1: return jitter_bri(img)
    elif choice == 2: return jitter_hue(img)
    elif choice == 3: return jitter_con(img)
    elif choice == 4: return jitter_sat(img)


def spatial_augment(img, gt=None):
    choice = random.choice([1, 2, 3, 4, 5])
    if choice == 1:
        img = VF.vflip(img)
        if gt is not None:
            gt = VF.vflip(gt)
    elif choice == 2:
        img = VF.hflip(img)
        if gt is not None:
            gt = VF.hflip(gt)
    elif choice == 3:
        img = VF.rotate(img, 90)
        if gt is not None:
            gt = VF.rotate(gt, 90)
    elif choice == 4:
        img = VF.rotate(img, 180)
        if gt is not None:
            gt = VF.rotate(gt, 180)
    elif choice == 5:
        img = VF.rotate(img, 270)
        if gt is not None:
            gt = VF.rotate(gt, 270)

    if gt is not None:
        return img, gt
    else:
        return img
    

### tests
if __name__ == 'main':
    pass ### TODO
