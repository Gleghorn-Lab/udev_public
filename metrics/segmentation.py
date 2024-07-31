from transformers import  EvalPrediction
from sklearn.metrics import f1_score, jaccard_score


def segmentation_metrics(p: EvalPrediction):
    seg_logits, labels = p
    seg_preds = seg_logits.argmax(axis=1).flatten() # (b, num_classes, H, W)
    labels = labels.flatten()

    # Calculate your desired metrics here
    accuracy = (seg_preds == labels).mean()
    f1 = f1_score(labels, seg_preds, average='weighted')
    iou = jaccard_score(labels, seg_preds, average='weighted')
    
    return {"accuracy": accuracy, "f1": f1, "iou": iou} 
