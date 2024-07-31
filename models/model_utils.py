### imports
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class VisionDropout(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(VisionDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def view_variable(x):
    # Variable the firt element of a batch in the middel of a pytorch model
    plt.imshow(x[0].detach().cpu().numpy())
    plt.show()


def load_from_weight_path(model, username='lhallee', weight_path=None, token=None):
    """
    Loads weights to the input model
    inputs
    model - pytorch model class (HF)
    username - huggingface username
    weight_path - where the weights are
    token - huggingface read token
    returns
    model - pytorch model class (HF)
    """
    if weight_path != None:
        if username in weight_path:
            model = model.from_pretrained(weight_path, token=token, config=model.config)
        else:
            try:
                model.load_state_dict(torch.load(weight_path)) # for torch
            except:
                from safetensors.torch import load_model
                load_model(model, weight_path) # for safetensors
        print(f'Loaded from {weight_path}')
    return model


def add_new_tokens(model, tokenizer, tokens):
    """
    Adds "tokens" as new tokens, seeds with CLS token
    inputs
    model - pytorch model class (HF)
    tokenizer - tokenizer class(HF)
    tokens - list of new tokens
    returns
    pytorch model (extended)
    tokenizer (extended)
    """
    with torch.no_grad():
        model.resize_token_embeddings(len(tokenizer) + len(tokens))
        # Add new tokens to the tokenizer
        added_tokens = {'additional_special_tokens' : tokens}
        tokenizer.add_special_tokens(added_tokens)
        # Seed the embedding with the [CLS] token embedding
        try:  
            cls_token_embedding = model.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
            for token in tokens:
                model.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()
        except AttributeError:
            cls_token_embedding = model.esm.embeddings.word_embeddings.weight[tokenizer.cls_token_id, :].clone()
            for token in tokens:
                model.esm.embeddings.word_embeddings.weight[tokenizer._convert_token_to_id(token), :] = cls_token_embedding.clone()
    return model, tokenizer


def count_parameters(model):
    """
    Counts model parameters
    input
    model - pytorch model
    output
    print statements
    """
    scale = 1e6
    total_params = sum(p.numel() for p in model.parameters()) / scale
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / scale
    total_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad) / scale
    # Count half precision parameters
    total_half_precision_params = sum(p.numel() for p in model.parameters()
                                      if (p.dtype is torch.bfloat16 or p.dtype is torch.float16)) / scale
    print(f'Total Parameters: {total_params}')
    print(f'Trainable Parameters: {total_trainable_params}')
    print(f'Frozen Parameters: {total_frozen_params}')
    print(f'Half Precision Parameters: {total_half_precision_params}')


def plot_expert_weight_differences(orig, trained, title, img_scale=1, power=3, save=False):
    """
    For analyzing changed weights during MoE extension
    inputs
    orig - pytorch model (HF)
    trained - pytorch model (HF)
    title - plot title
    img_scale - larger makes plot smaller
    power - exaggerates the visual difference between experts
    save - save plot or not
    returns
    plots and saves
    """
    num_layers = orig.config.num_hidden_layers
    num_experts = orig.config.num_local_experts

    # Set up a large plot with subplots for each expert layer
    fig, axes = plt.subplots(num_layers, num_experts, figsize=(num_experts * 12, num_layers * 8))
    fig.suptitle(title, fontsize=60)

    if num_layers * num_experts > 1:
        axes = axes.flatten()

    for name, param1 in orig.named_parameters():
        if 'experts' in name and 'bias' not in name:
            param2 = dict(trained.named_parameters())[name]
            layer = int(name.split('layer')[1].split('.')[1])
            expert = int(name.split('experts')[1].split('.')[1])
            # Compute squared differences
            squared_diff = (param1 - param2) ** power

            # Reshape to 2D for visualization
            if squared_diff.dim() > 2:
                squared_diff = squared_diff.view(squared_diff.size(0), -1)

            # Convert to numpy array for plotting
            squared_diff_np = squared_diff.cpu().detach().numpy()
            if img_scale != 1:
                from PIL import Image
                img = Image.fromarray(squared_diff_np)
                img = img.resize((img.size[0]//img_scale, img.size[1]//img_scale))
                squared_diff_np = np.array(img)
            ax_idx = layer * num_experts + expert
            ax = axes[ax_idx]
            im = ax.imshow(squared_diff_np, cmap='coolwarm', aspect='auto')
            ax.set_title(f'Layer {layer+1} Expert {expert+1}')

            # Remove x and y ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar and adjust layout
    plt.subplots_adjust(top=0.9)  # Adjust top to accommodate main title

    if save:
        plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()



def weight_compare(orig, trained, title, scale=False, save=False):
    """
    Plots the change in weights during training
    orig - pytorch model (HF)
    trained - pytorch model (HF)
    title - plot title
    scale - apply log scale or not
    save - save or not
    returns
    plots and saves
    """
    mse_values = {}
    layer_counts = {}

    for name1, param1 in orig.named_parameters():
        param2 = dict(trained.named_parameters())[name1]
        # Mapping layers to groups
        if 'word_embeddings' in name1:
            group = 'Token Embedding'
        elif 'attention' in name1:
            group = 'Attention'
        elif 'gate' in name1:
            group = 'Router'
        elif 'experts' in name1:
            # Group experts by layer number
            layer_number = name1.split('layer')[1].split('.')[1]
            group = 'Expert Layer ' + layer_number
        else:
            continue

        mse = torch.nn.functional.mse_loss(param1, param2).item()

        if group not in mse_values:
            mse_values[group] = 0
            layer_counts[group] = 0

        mse_values[group] += mse
        layer_counts[group] += 1

    for group in mse_values:
        mse_values[group] /= layer_counts[group]

    sorted_groups = sorted(mse_values.keys(), key=lambda k: mse_values[k], reverse=True)
    sorted_mse_values = [mse_values[k] for k in sorted_groups]
    # Scale the figure size with how many groups there are
    plt.figure(figsize=(12, int(len(sorted_groups)/2)))
    plt.barh(sorted_groups, sorted_mse_values, color='skyblue')
    if scale: # Using a log scale for better visualization
        plt.xscale('log')
    plt.xlabel('Average MSE', fontsize=9)
    plt.ylabel('Weight Type', fontsize=9)
    plt.title(title, fontsize=10)
    plt.gca().invert_yaxis()
    if save:
        plt.savefig(f'{title}.png', dpi=450, facecolor='white')
    plt.show()


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


### tests
if __name__ == 'main':
    pass
