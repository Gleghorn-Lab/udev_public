import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.special import expit


def singlelabel_confidences(logits, labels):
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    correct_preds = preds == labels
    confidences = probs[np.arange(len(probs)), preds]
    positives = confidences[correct_preds]
    negatives = confidences[~correct_preds]
    return positives, negatives, confidences, correct_preds


def multilabel_confidences(logits, labels):
    probs = expit(logits)
    preds = probs >= 0.5
    ones = preds == 1
    correct_preds = ones & (preds == labels)
    incorrect_preds = ones & ~correct_preds
    positives = probs[correct_preds]
    negatives = probs[incorrect_preds]
    return positives, negatives, probs, correct_preds


def confidence_plot(positives, negatives):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(positives, bins='auto', alpha=0.5, label='Correct')
    ax.hist(negatives, bins='auto', alpha=0.5, label='Incorrect')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution')
    ax.legend()
    return fig, ax
    

def sub_plot(positives, negatives, ax, title='plot', correlation=None, pval=None):
    ax.hist(positives, bins='auto', alpha=0.5, label='Correct')
    ax.hist(negatives, bins='auto', alpha=0.5, label='Incorrect')
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('Confidence')
    if correlation is not None and pval is not None:
        ax.text(0.05, 0.95, f"ρ={correlation:.2f}\np={pval:.2e}", transform=ax.transAxes, ha='left', va='top')


def plot_all_confidences(positives_list, negatives_list, titles, nrows, ncols, wscale=5, vscale=10, save_path=None):
    figsize = (wscale * ncols, vscale * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j < len(positives_list):
                positives = positives_list[i * ncols + j]
                negatives = negatives_list[i * ncols + j]
                confidences = np.concatenate((positives, negatives))
                correct_preds = np.concatenate((np.ones_like(positives), np.zeros_like(negatives)))
                correlation, pval = spearmanr(confidences, correct_preds)
                sub_plot(positives, negatives, axes[i, j], titles[i * ncols + j], correlation, pval)
            else:
                axes[i, j].axis('off')  # Hide empty subplots

    # Remove y-axis labels and ticks for all subplots
    for ax in axes.flat:
        ax.set_ylabel('')
        ax.set_yticks([])

    # Find the first empty subplot space
    empty_subplot_index = len(positives_list)
    legend_subplot = axes.flat[empty_subplot_index]

    # Add the legend to the empty subplot space
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_subplot.legend(handles, labels, loc='center')
    legend_subplot.axis('off')

    axes[0, 0].set_ylabel('Frequency')
    axes[-1, 0].set_ylabel('Frequency')

    plt.tight_layout(h_pad=1.5)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()