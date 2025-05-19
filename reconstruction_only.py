import matplotlib
matplotlib.use('TkAgg') 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_size = 28 * 28
latent_dim = 64
batch_size = 128
learning_rate = 1e-3
num_epochs = 30

# Dataset and DataLoader
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(-1, 1, 28, 28)
        return reconstructed, latent

# Initialize model, loss, optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        outputs, _ = model(images)
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("\nTraining complete. Generating reconstruction visualization...\n")

def plot_images(original, reconstructed, n=6):
    """ Plot original and reconstructed images side by side """
    plt.figure(figsize=(12, 4))
    for i in range(n):
        # Original images
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Reconstructed images
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)  

# Select a batch of test images
test_images, _ = next(iter(train_loader))
test_images = test_images.to(device)

# Generate reconstructions
with torch.no_grad():
    reconstructed, _ = model(test_images)

# Plot original vs reconstructed images
plot_images(test_images.cpu(), reconstructed.cpu())
