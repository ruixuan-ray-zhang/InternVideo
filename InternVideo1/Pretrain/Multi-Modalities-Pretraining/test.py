import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def visualize_attention_map(attention_weights, save_path=None):

    # Plot the attention map
    plt.figure(figsize=(5, 5))
    plt.imshow(attention_weights[0].transpose(), cmap='hot', interpolation='nearest')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.colorbar()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

attention_weights = torch.randn(1, 10, 10).numpy()  # (key_length, batch_size, embedding_dim)

# Visualize the attention map and save the plot
save_path = 'attention_map.png'
visualize_attention_map(attention_weights, save_path=save_path)