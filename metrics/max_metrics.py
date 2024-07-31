import torch


def calculate_max_metrics(ss, labels, cutoff):
    ss, labels = ss.float(), labels.float()
    tp = torch.sum((ss >= cutoff) & (labels == 1.0))
    fp = torch.sum((ss >= cutoff) & (labels == 0.0))
    fn = torch.sum((ss < cutoff) & (labels == 1.0))
    precision_denominator = tp + fp
    precision = torch.where(precision_denominator != 0, tp / precision_denominator, torch.tensor(0.0))
    recall_denominator = tp + fn
    recall = torch.where(recall_denominator != 0, tp / recall_denominator, torch.tensor(0.0))
    f1 = torch.where((precision + recall) != 0, (2 * precision * recall) / (precision + recall), torch.tensor(0.0))
    return f1, precision, recall


def max_metrics(ss, labels, increment=0.01):
    ss = torch.clamp(ss, -1.0, 1.0)
    min_val = ss.min().item()
    max_val = 1
    if min_val >= max_val:
        min_val = 0
    cutoffs = torch.arange(min_val, max_val, increment)
    metrics = [calculate_max_metrics(ss, labels, cutoff.item()) for cutoff in cutoffs]
    f1s = torch.tensor([metric[0] for metric in metrics])
    precs = torch.tensor([metric[1] for metric in metrics])
    recalls = torch.tensor([metric[2] for metric in metrics])
    valid_f1s = torch.where(torch.isnan(f1s), torch.tensor(-1.0), f1s)  # Replace NaN with -1 to ignore them in argmax
    max_index = torch.argmax(valid_f1s)
    return f1s[max_index].item(), precs[max_index].item(), recalls[max_index].item(), cutoffs[max_index].item()