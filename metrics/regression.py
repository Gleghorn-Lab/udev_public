import numpy as np
from transformers import EvalPrediction
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def spearman(probs, correct_preds):
    correlation, pval = spearmanr(probs.flatten(), correct_preds.flatten())
    print(f'Spearman: {correlation:.3f}, {pval:.3E}')


def get_residuals(preds, labels):
    return np.abs(labels.flatten() - preds.flatten())


def compute_metrics_regression(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids[1] if isinstance(p.label_ids, tuple) else p.label_ids

    logits = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    r2 = r2_score(labels, logits)
    spearman_rho, pval = spearmanr(logits, labels)
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)

    return {
        'r_squared': r2,
        'spearman_rho': spearman_rho,
        'pval': pval,
        'mse': mse,
        'mae': mae
    }
