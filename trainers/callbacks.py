import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.metrics import f1_score, jaccard_score
from typing import *
from transformers import  EvalPrediction
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainingArguments, TrainerControl


class TopkTallyCallback(TrainerCallback):
    """
    Plots the MoE experts usage at every model save
    """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs['model']
        try:
            tally = model.aux_loss.tally.detach().cpu().numpy()
            model.aux_loss.reset_tally()
        except:
            tally = model.expert_loss.tally.detach().cpu().numpy()
            model.expert_loss.reset_tally()
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(tally.shape[0]), tally)
        plt.xlabel('Expert Index')
        plt.ylabel('Tally of Topk Chosen Results')
        plt.title(f'Topk Tally at Global Step {state.global_step}')
        plt.savefig(f'topk_tally_{state.global_step}.png')
        plt.close()


def prepare_attention(attentions):
    if attentions[0].ndim == 4: # (b, h, H, W)
        attns = []
        for att in attentions:
            attns.append(att.squeeze(1))
    else: # (b, H, W)
        attns = attentions
    attns = torch.cat(attns) # (num_attentions, img_size, img_size)
    attns = attns.mean(dim=0) # (img_size, img_size)
    return attns


class AttentionMapCallback(TrainerCallback):
    def __init__(self, eval_dataset, id2label, alpha=0.8):
        self.eval_dataset = eval_dataset
        self.id2label = id2label
        self.alpha = alpha

    def on_evaluate(self, args, state, control, **kwargs):
        sample_idx = random.randint(0, len(self.eval_dataset) - 1)
        sample = self.eval_dataset[sample_idx]
        image = torch.tensor(sample['img']).unsqueeze(0).to(args.device)
        label = torch.tensor(sample['labels'])

        with torch.no_grad():
            outputs = kwargs['model'](image, output_attentions=True)
            logits = outputs.logits.cpu()
            attentions = outputs.attentions
        attentions = prepare_attention(attentions).cpu()

        predicted_class = logits.argmax(dim=1).item()

        image = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
        grayscale_image = np.mean(image, axis=-1)

        # Normalize the attention map to the range [0, 1]
        attention_map = attentions.numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

        # Create a jet colormap
        colormap = cm.jet

        # Map the attention values to RGB colors using the jet colormap
        attention_color = colormap(attention_map)

        # Overlay the attention map on the grayscale image with transparency
        overlaid_image = grayscale_image[..., np.newaxis] * (1 - self.alpha) + attention_color[..., :3] * self.alpha

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(image)
        axs[0].set_title(f"Original Image\nTrue Label: {self.id2label[label.item()]}, Predicted: {self.id2label[predicted_class]}")
        axs[0].axis('off')

        axs[1].imshow(overlaid_image)
        axs[1].set_title("Attention Map Overlaid on Grayscale Image")
        axs[1].axis('off')

        # Add a colorbar to represent the attention values
        cbar = fig.colorbar(cm.ScalarMappable(cmap=colormap), ax=axs[1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention')

        plt.tight_layout()
        plt.show()


def visualize_segmentation(image: np.ndarray, pred: np.ndarray, label: np.ndarray, save_path: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Print shapes for debugging
    print(f"Original Image Shape: {image.shape}")
    print(f"Prediction Shape: {pred.shape}")
    print(f"Ground Truth Shape: {label.shape}")

    # Plot original image
    if image.ndim == 3:
        image = image / 255.0 if image.max() > 1.0 else image  # Normalize to [0, 1] if necessary
        axes[0].imshow(image)  # Image already in (H, W, C) format
    else:
        axes[0].imshow(image, cmap='gray')  # Assuming grayscale image
    axes[0].set_title('Original Image')
    
    # Plot prediction
    axes[1].imshow(pred.squeeze(), cmap='gray', vmin=0, vmax=1)  # Adjust cmap and value range as needed for predictions
    axes[1].set_title('Prediction')
    
    # Plot ground truth
    axes[2].imshow(label.squeeze(), cmap='gray', vmin=0, vmax=1)  # Adjust cmap and value range as needed for ground truth
    axes[2].set_title('Ground Truth')
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)  # Close the figure to prevent displaying it in console
    return save_path


output_dir = '.'


def segmentation_metrics(p: EvalPrediction, dataset: List[Dict]) -> Dict[str, float]:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    seg_logits, labels = p
    seg_logits = torch.tensor(seg_logits)  # Convert seg_logits to a PyTorch tensor if it's a numpy array
    seg_preds = seg_logits.argmax(axis=1)  # (b, H, W)
    labels = torch.tensor(labels.flatten(), dtype=torch.long)
    
    # Calculate your desired metrics here
    accuracy = (seg_preds.flatten() == labels).float().mean().item()  # Calculate accuracy correctly
    f1 = f1_score(labels.cpu().numpy(), seg_preds.flatten().cpu().numpy(), average='weighted')
    iou = jaccard_score(labels.cpu().numpy(), seg_preds.flatten().cpu().numpy(), average='weighted')

    # Print shapes for debugging
    print(f"seg_logits shape: {seg_logits.shape}")
    print(f"seg_preds shape: {seg_preds.shape}")
    print(f"labels shape: {labels.shape}")

    # Generate visualizations for each example in the batch
    visualizations = []
    
    for i, data in enumerate(dataset):  # Iterate through each sample in the dataset
        if 'img' in data and 'labels' in data:
            image = data['img']
            label = data['labels']

            # Convert image and label to numpy arrays
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to numpy array (H, W, C)
            else:
                print(f"Expected 'img' to be a torch.Tensor, got {type(image)}")
                continue

            if isinstance(label, torch.Tensor):
                label_np = label.permute(1, 2, 0).cpu().numpy() if label.ndim == 3 else label.cpu().numpy()  # (H, W, C) or (H, W)
            else:
                print(f"Expected 'labels' to be a torch.Tensor, got {type(label)}")
                continue

            # Ensure seg_preds has the correct shape and content
            if isinstance(seg_preds, torch.Tensor):
                if seg_preds.dim() == 3:  # If seg_preds has shape (H, W) instead of (b, H, W)
                    seg_preds_np = seg_preds.cpu().numpy()
                else:
                    seg_preds_np = seg_preds[i].cpu().numpy().squeeze()  # Accessing the i-th sample in seg_preds and remove extra dimensions
            else:
                print(f"Expected 'seg_preds' to be a torch.Tensor, got {type(seg_preds)}")
                continue
            
            save_path = os.path.join(output_dir, f"visualization_sample_{i}.png")
            print(f"Saving visualization {i} to {save_path}")  # Debug statement before saving
            file_path = visualize_segmentation(image_np, seg_preds_np, label_np, save_path)
            visualizations.append(file_path)
            
        else:
            print(f"Keys 'img' or 'labels' not found in dataset item at index {i}")

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, IoU: {iou:.4f}")

    return {"accuracy": accuracy, "f1": f1, "iou": iou, "visualizations": visualizations}