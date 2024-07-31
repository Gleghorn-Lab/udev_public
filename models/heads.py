import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import ImageClassifierOutput, SemanticSegmenterOutput
from transformers.activations import get_activation


class VitSegmentationHead(nn.Module):
    def __init__(self, img_size=224, patch_size=16, hidden_size=768, output_channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.ConvTranspose2d(hidden_size, output_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # (B, num_patches, hidden_size)
        x = x.transpose(1, 2).view(-1, self.hidden_size,
                                   self.img_size // self.patch_size,
                                   self.img_size // self.patch_size)
        return self.proj(x)  # (B, num_classes, H, W)
    

class UnetClassificationHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(args.dim, args.num_classes)

    def forward(self, x): # (bs, c, L, L) t
        x = self.fc(x) # (bs, c, L, num_classes)
        x = x.permuate(0, 3, 1, 2) # (bs, num_classes, c, L)
        x = self.avg_pool(x) # (bs, num_classes, 1, 1)
        x = x.flatten() # (bs, num_classes)
        return x


class ImageClassificationLossFromLogits(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.problem_type = args.problem_type

    def forward(self, logits, labels=None, att=None): # (bs, num_classes), (bs, 1)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.problem_type is None:
                if self.num_classes == 1:
                    self.problem_type = "regression"
                elif self.num_classes > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_classes == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if att != None:
            attentions = att
        else:
            attentions = None

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=attentions,
        )


class SegmentationLossFromLogits(nn.Module):
    def __init__(self, num_classes=2, ignore_index=-100):
        super().__init__()
        if num_classes > 1:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes

    def forward(self, logits, labels=None): # (bs, num_classes, H, W), (bs, H, W)
        bs, _, H, W = logits.size()
        loss = None
        if labels is not None:
            if self.num_classes == 1:
                logits = logits.squeeze(1)
                labels = labels.float()
            else:
                labels = labels.long()           
            loss = self.criterion(logits, labels.view(bs, H, W))

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

class MeanPoolingHead(nn.Module):
    """Head for sentence-level classification tasks with mean pooling."""
    def __init__(self, hidden_size, output_dim, act='silu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, output_dim)
        self.act = get_activation(act)
        self.out = nn.Linear(output_dim, output_dim)

    def forward(self, state):
        vec = state.mean(dim=1)
        vec = self.act(self.dense(vec))
        vec = self.out(vec)
        return vec



class CLSPoolerHead(nn.Module):
    """Head for sentence-level classification tasks with cls pooling."""
    def __init__(self, hidden_size, output_dim, act='silu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, output_dim)
        self.act = get_activation(act)
        self.out = nn.Linear(output_dim, output_dim)
    
    def forward(self, state):
        vec = state[:, 0]
        vec = self.act(self.dense(vec))
        vec = self.out(vec)
        return vec


class LanguageModelingHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, eps=1e-05, act='silu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.act = get_activation(act)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, features):
        x = self.dense(features)
        x = self.act(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    print('Testing SegmentationLossFromLogits')
    batch_size, H, W = 2, 224, 224
    num_classes = 2
    tester = SegmentationLossFromLogits(num_classes=num_classes)
    logits, labels = torch.rand(batch_size, num_classes, H, W), torch.randint(0, num_classes, (batch_size, H, W))
    output = tester(logits, labels)
    print(output.loss)

    num_classes = 1
    tester = SegmentationLossFromLogits(num_classes=num_classes)
    logits, labels = torch.rand(batch_size, num_classes, H, W), torch.randint(0, num_classes, (batch_size, H, W))
    output = tester(logits, labels)
    print(output.loss)