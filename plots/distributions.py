import numpy as np
import matplotlib.pyplot as plt


def normal_masking_dists(mean=0.7, std=0.1):
    np.random.seed(42)
    samples = 100000
    prob = np.random.normal(mean, std, samples)
    clipped_prob = np.clip(prob, 0.05, 1)
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(clipped_prob, bins=20, range=(0, 1), edgecolor='black')
    plt.title('Distribution of Clipped Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    # Add vertical line for mean
    plt.axvline(np.mean(clipped_prob), color='r', linestyle='dashed', linewidth=2, label=f'Mean = {np.mean(clipped_prob):.3f}')
    # Add text for standard deviation
    plt.text(0.05, 0.95, f'Standard Deviation = {np.std(clipped_prob):.3f}', 
            transform=plt.gca().transAxes, verticalalignment='top')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    # Print some statistics
    print(f"Mean: {np.mean(clipped_prob):.3f}")
    print(f"Standard Deviation: {np.std(clipped_prob):.3f}")
    print(f"Min: {np.min(clipped_prob):.3f}")
    print(f"Max: {np.max(clipped_prob):.3f}")


if __name__ == '__main__':
    normal_masking_dists(0.3, 0.3)
