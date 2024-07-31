import numpy as np
import torch
import torch.nn.functional as F
from transformers import EvalPrediction
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    confusion_matrix
)
from .max_metrics import max_metrics


def compute_metrics_multi_label_classification(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    preds = np.array(preds)
    labels = np.array(labels)

    preds = torch.tensor(preds)
    y_true = torch.tensor(labels, dtype=torch.int)

    probs = F.softmax(preds, dim=-1)
    y_pred = (probs > 0.5).int()

    f1, prec, recall, thres = max_metrics(probs, y_true)
    accuracy = accuracy_score(y_pred.flatten(), y_true.flatten())
    hamming = hamming_loss(y_pred.flatten(), y_true.flatten())
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': prec,
        'recall': recall,
        'hamming_loss': hamming,
        'threshold': thres
    }


def compute_metrics_single_label_classification(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    preds = torch.tensor(np.array(preds))
    y_true = torch.tensor(np.array(labels), dtype=torch.int)

    if y_true.size() == preds.size():
        probs = F.softmax(preds, dim=-1)
        y_pred = (probs > 0.5).int().flatten()
    else:
        y_pred = preds.argmax(dim=-1).flatten()

    y_true = y_true.flatten()
    valid_indices = y_true != -100

    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }
