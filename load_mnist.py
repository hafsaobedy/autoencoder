import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transform: convert images to tensors only (no normalization for better autoencoder compatibility)
transform = transforms.ToTensor()

# Load MNIST training dataset from torchvision
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Dataset analysis
print(f"Number of training samples: {len(train_dataset)}")

# Visualize the first 6 images with labels
fig, axs = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    image, label = train_dataset[i]
    axs[i].imshow(image.squeeze(), cmap='gray')
    axs[i].set_title(f"Label: {label}")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_samples.png')  # Save the figure as mnist_samples.png
plt.show()
