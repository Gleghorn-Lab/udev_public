import torch
import torch.nn.functional as F
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


def plot_embeddings_pca(embeddings, labels, title):
    """
    Plots the PCA embeddings of the given embeddings, labeled by the first element of the labels list.
    
    Args:
    embeddings (list of tensors): The embeddings to plot.
    labels (list of lists): The labels for each embedding.
    title (str): The title of the plot.
    """
    embeddings_array = torch.stack(embeddings).numpy()
    pca_embeddings = PCA(n_components=2).fit_transform(embeddings_array)

    first_labels = [label_list[0] for label_list in labels] # get first EC
    unique_labels = sorted(list(set(first_labels)))
    color_map = plt.cm.get_cmap('viridis', len(unique_labels))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        mask = [first_label == label for first_label in first_labels]
        ax.scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], c=[color_map(i)], label=label)
    
    title = title + ' PCA'
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    title = title.replace(' ', '_')
    plt.savefig('plots/' + title + '.png', dpi=300)
    plt.show()


def plot_embeddings_tsne(embeddings, labels, title):
    """
    Plots the t-SNE embeddings of the given embeddings, labeled by the first element of the labels list.
    
    Args:
    embeddings (list of tensors): The embeddings to plot.
    labels (list of lists): The labels for each embedding.
    title (str): The title of the plot.
    """
    embeddings_array = torch.stack(embeddings).numpy()
    tsne_embeddings = TSNE(n_components=2, random_state=42).fit_transform(embeddings_array)

    first_labels = [label_list[0] for label_list in labels]  # get first EC
    unique_labels = sorted(list(set(first_labels)))
    color_map = plt.cm.get_cmap('viridis', len(unique_labels))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        mask = [first_label == label for first_label in first_labels]
        ax.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1], c=[color_map(i)], label=label)
    
    title = title + ' TSNE'
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    title = title.replace(' ', '_')
    plt.savefig('plots/' + title + '.png', dpi=300)
    plt.show()


def get_data(df):
    try:
        ecs = df['EC number'].tolist()
    except:
        ecs = df['EC'].tolist()
    try:
        seqs = df['Sequence'].tolist()
    except:
        seqs = df['Sequences'].tolist()
    ecs = [ec[0] for ec in ecs]
    return ecs, seqs


def load_embeddings(seqs, db_file):
    embeddings = []
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for seq in tqdm(seqs, desc='Loading from disk'):
        result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
        row = result.fetchone()
        emb_data = row[0]
        emb = F.normalize(torch.tensor(np.frombuffer(emb_data, dtype=np.float32)), dim=-1, p=2)
        embeddings.append(emb)
    conn.close()
    return embeddings


def main(data_path, db_file, title):
    df = pd.read_csv(data_path, delimiter='\t')
    ecs, seqs = get_data(df)
    embs = load_embeddings(seqs, db_file)
    plot_embeddings_pca(embs, ecs, title)
    plot_embeddings_tsne(embs, ecs, title)


if __name__ == '__main__':
    data_path = './data/' + input('Data path: ')
    db_file = './data/' + input('DB path: ')
    title = input('Name the plot: ')
    main(data_path, db_file, title)
