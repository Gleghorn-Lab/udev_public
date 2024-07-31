import torch
import torch.nn.functional as F
import csv
import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")


### From CLEAN
def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i


def write_max_sep_choices(df, csv_name, first_grad=True, use_max_grad=False):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df.iloc[i]
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_true_labels(file_name):
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label


def get_pred_labels(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label


def get_pred_probs(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_probs = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        probs = torch.zeros(len(preds_with_dist))
        count = 0
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = float(pred_ec_dist.split(":")[1].split("/")[1])
            probs[count] = ec_i
            #preds_ec_lst.append(probs)
            count += 1
        # sigmoid of the negative distances 
        probs = (1 - torch.exp(-1/probs)) / (1 + torch.exp(-1/probs))
        probs = probs/torch.sum(probs)
        pred_probs.append(probs)
    return pred_probs


def get_eval_metrics(pred_label, pred_probs, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    label_pos_dict = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
        labels, probs = pred_label[i], pred_probs[i]
        for label, prob in zip(labels, probs):
            if label in all_label:
                pos = label_pos_dict[label]
                pred_m_auc[i, pos] = prob
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, roc, acc


def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
        
    return label_pos_dict


### Custom
def plot_embeddings_pca(embeddings, labels):
    embeddings_array = torch.stack(embeddings).numpy()
    pca_embeddings = PCA(n_components=2).fit_transform(embeddings_array)

    first_labels = [label_list[0] for label_list in labels]  # get first EC
    unique_labels = sorted(list(set(first_labels)))
    
    # Create a color list using a built-in colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        mask = [first_label == label for first_label in first_labels]
        ax.scatter(pca_embeddings[mask, 0], pca_embeddings[mask, 1], c=[colors[i]], label=label)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    plt.tight_layout()
    plt.show()


def create_dicts(df):
    ids = df['Entry'].tolist()
    ecs = df['EC number'].tolist()
    seqs = df['Sequence'].tolist()

    id_ec = dict(zip(ids, ecs))
    id_seq = dict(zip(ids, seqs))
    
    ec_id = {}
    for row in df.iterrows():
        if row[0] > 0: # tuple, 0 is index
            info = row[1]
            ids = info.iloc[0]
            ecs = info.iloc[1].split(';')
            for ec in ecs:
                ec_id.setdefault(ec, set()).add(ids)
    return id_ec, id_seq, ec_id


def load_embeddings(seqs, path, split=None):
    embeddings = []
    if '.db' in path:
        conn = sqlite3.connect(path)
        c = conn.cursor()
        for seq in tqdm(seqs, desc='Loading from disk'):
            result = c.execute("SELECT embedding FROM embeddings WHERE sequence=?", (seq,))
            row = result.fetchone()
            emb_data = row[0]
            embeddings.append(torch.tensor(np.frombuffer(emb_data, dtype=np.float32)))
        conn.close()
    else:
        data = load_dataset(path, split=split)
        data_seqs = data['seqs']
        data_embeds = data['vectors']
        embed_dict = dict(zip(data_seqs, data_embeds))
        for seq in tqdm(seqs, desc='Loading from HF'):
            embeddings.append(torch.tensor(embed_dict[seq]).squeeze(0))
    print(embeddings[0][:5])
    return embeddings


def get_cluster_centers(id_emb, ec_id):
    cluster_centers = {}
    for ec, ids in ec_id.items():
        embeddings = [id_emb[id] for id in ids]
        cluster_centers[ec] = torch.stack(embeddings).mean(dim=0).detach().cpu()
    return cluster_centers

"""
### Naive version
def calc_dist_map(ec_center, test_id_emb):
    dist_map = {}
    for id, emb in tqdm(test_id_emb.items(), desc='Calculating distance'):
        dist_map[id] = {}
        for ec, center in ec_center.items():
            dist_map[id][ec] = torch.dist(emb, center).item()
    return dist_map
"""

def calc_dist_map(ec_center, test_id_emb): # batched version
    ec_centers = torch.stack(list(ec_center.values()))
    test_embeddings = torch.stack(list(test_id_emb.values()))
    distances = torch.cdist(test_embeddings, ec_centers)
    ec_names = list(ec_center.keys())
    test_ids = list(test_id_emb.keys())
    dist_map = {test_id: {ec: dist.item() for ec, dist in zip(ec_names, test_dists)} 
                for test_id, test_dists in zip(test_ids, distances)}

    return dist_map


def calc_centers(train_file, db_file, split=None):
    train = pd.read_csv(train_file, delimiter='\t')
    _, train_id_seq, train_ec_id = create_dicts(train)
    train_ids = list(train_id_seq.keys())
    train_seqs = list(train_id_seq.values())

    train_embs = load_embeddings(train_seqs, db_file, split=split)
    train_id_emb = dict(zip(train_ids, train_embs))
    ec_center = get_cluster_centers(train_id_emb, train_ec_id)
    return ec_center


def calculate_metrics(eval_df, test_name): # no csv
    ensure_dirs("./results")
    out_filename = "results/" +  test_name
    write_max_sep_choices(eval_df, out_filename, use_max_grad=True)
    pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
    pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
    true_label, all_label = get_true_labels('./data/' + test_name)
    pre, rec, f1, roc, acc = get_eval_metrics(pred_label, pred_probs, true_label, all_label)
    print(f"############ EC calling results using maximum separation {test_name} ############")
    print('-' * 75)
    print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
        f'>>> precision: {pre:.3} | recall: {rec:.3}'
        f'| F1: {f1:.3} | ACC: {acc:.3} | AUC: {roc:.3} ')
    print('-' * 75)


def infer_max_sep(train_name, path, test_names, split=None, plot=False):
    train_file = f'./data/{train_name}.csv'
    ec_center = calc_centers(train_file, path, split=split)
    for test_name in test_names:
        test_file = f'./data/{test_name}.csv'
        ### Get test data
        test = pd.read_csv(test_file, delimiter='\t')
        test_id_ec, test_id_seq, _ = create_dicts(test)
        test_ids = list(test_id_seq.keys())
        test_seqs= list(test_id_seq.values())
        test_ec = list(test_id_ec.values())

        ### Load embeddings
        test_embs = load_embeddings(test_seqs, path, split=split)
        test_id_emb = dict(zip(test_ids, test_embs))

        if plot:
            plot_embeddings_pca(test_embs, test_ec)

        ### To dataframe
        dist_map = calc_dist_map(ec_center, test_id_emb)
        eval_df = pd.DataFrame.from_dict(dist_map)

        ### Record
        calculate_metrics(eval_df, test_name)


if __name__ == '__main__':
    PLMS = {
        'asm35_exp':'',
        'asm35_red':'',
        #"esm2_8": "facebook/esm2_t6_8M_UR50D",
        #"esm2_35": "facebook/esm2_t12_35M_UR50D",
        #"esm2_150": "facebook/esm2_t30_150M_UR50D",
        #"esm2_650": "facebook/esm2_t33_650M_UR50D",
        #"ankh_base": "ankh-base",
        #"ankh_large": "ankh-large",
        #"protvec": "lhallee/ProteinVec",
    }
    for model_name, plm_path in PLMS.items():
        print(f'Results for {model_name}')
        infer_max_sep('split100', 'lhallee/plm_embeddings', ['new', 'price', 'halogenase'], split=model_name, plot=False)
