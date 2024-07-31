import torch
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics_mlm(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    logits = torch.tensor(np.array(logits))
    labels = torch.tensor(np.array(labels), dtype=torch.int)

    y_pred = logits.argmax(dim=-1).flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }


def compute_metrics_mlm_from_pred(p: EvalPrediction):
    # from preds (argmaxed) instead of logits
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    preds = torch.tensor(np.array(preds))
    labels = torch.tensor(np.array(labels), dtype=torch.int)

    y_pred = preds.flatten()
    y_true = labels.flatten()
    valid_indices = y_true != -100
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
