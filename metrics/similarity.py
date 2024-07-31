import torch
import torch.nn.functional as F
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score
from itertools import combinations_with_replacement
from .max_metrics import max_metrics


def compute_metrics_sentence_similarity(p: EvalPrediction):
    preds = p.predictions
    labels = p.label_ids[-1]
    emb_a, emb_b = preds[0], preds[1]
    # Convert embeddings to tensors
    emb_a_tensor = torch.tensor(emb_a)
    emb_b_tensor = torch.tensor(emb_b)
    labels_tensor = torch.tensor(labels)

    # Compute cosine similarity between the embeddings
    cosine_sim = F.cosine_similarity(emb_a_tensor, emb_b_tensor)
    # Compute max metrics
    f1, prec, recall, thres = max_metrics(cosine_sim, labels_tensor)
    # Compute accuracy based on the threshold found
    predictions = (cosine_sim > thres).float()
    acc = accuracy_score(predictions.flatten().numpy(), labels.flatten())
    # Compute the mean absolute difference between cosine similarities and labels
    dist = torch.mean(torch.abs(cosine_sim - labels_tensor)).item()
    # Return a dictionary of the computed metrics
    return {
        'accuracy': acc,
        'f1_max': f1,
        'precision_max': prec,
        'recall_max': recall,
        'threshold': thres,
        'distance': dist,
    }


def compute_metrics_sentence_similarity_test(p: EvalPrediction):
    preds = p.predictions
    emb_a, emb_b = preds[0], preds[1]
    # Convert embeddings to tensors
    emb_a_tensor = torch.tensor(emb_a)
    emb_b_tensor = torch.tensor(emb_b)

    # Compute cosine similarity between the embeddings
    cosine_sim = F.cosine_similarity(emb_a_tensor, emb_b_tensor)
    # Compute average cosine similarity
    avg_cosine_sim = torch.mean(cosine_sim).item()

    # Compute Euclidean distance between the embeddings
    euclidean_dist = torch.norm(emb_a_tensor - emb_b_tensor, p=2, dim=1)
    # Compute average Euclidean distance
    avg_euclidean_dist = torch.mean(euclidean_dist).item()

    # Return a dictionary of the computed metrics
    return {
        'avg_cosine_similarity': avg_cosine_sim,
        'avg_euclidean_distance': avg_euclidean_dist,
    }


def compute_metrics_double(p: EvalPrediction):
    preds = p.predictions
    emb_a, emb_b = preds[0], preds[1]
    # Convert embeddings to tensors
    emb_a_tensor = torch.tensor(emb_a)
    emb_b_tensor = torch.tensor(emb_b)

    # Compute cosine similarity between all combinations of indices in the batch
    batch_size = emb_a_tensor.shape[0]
    pair_similarities = []
    non_pair_similarities = []

    for i, j in combinations_with_replacement(range(batch_size), 2):
        cosine_sim = F.cosine_similarity(emb_a_tensor[i], emb_b_tensor[j], dim=0).item()
        if i == j:
            pair_similarities.append(cosine_sim)
        else:
            non_pair_similarities.append(cosine_sim)

    # Compute average cosine similarity of pairs
    avg_pair_similarity = sum(pair_similarities) / len(pair_similarities) if pair_similarities else 0

    # Compute average cosine similarity of non-pairs
    avg_non_pair_similarity = sum(non_pair_similarities) / len(non_pair_similarities) if non_pair_similarities else 0

    # Compute the ratio of similarity between pairs vs. non-pairs
    similarity_ratio = abs(avg_pair_similarity / (avg_non_pair_similarity + 1e-8))

    # Return a dictionary of the computed metrics
    return {
        'avg_pair_similarity': avg_pair_similarity,
        'avg_non_pair_similarity': avg_non_pair_similarity,
        'similarity_ratio': similarity_ratio,
    }
