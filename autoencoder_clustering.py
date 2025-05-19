import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Windows plotting

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# Print model parameter count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

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

print("\nTraining complete. Extracting latent features...\n")

# Extract latent features
model.eval()
latent_vectors = []
labels = []

with torch.no_grad():
    for images, targets in train_loader:
        images = images.to(device)
        _, latent = model(images)
        latent_vectors.append(latent.cpu().numpy())
        labels.append(targets.numpy())

latent_vectors = np.vstack(latent_vectors)
labels = np.hstack(labels)

print("Latent features extracted. Performing clustering...\n")

# Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(latent_vectors)

# Evaluate clustering with traditional metrics
sil_score = silhouette_score(latent_vectors, cluster_labels)
db_score = davies_bouldin_score(latent_vectors, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")

# Evaluate clustering accuracy with ARI and NMI
ari = adjusted_rand_score(labels, cluster_labels)
nmi = normalized_mutual_info_score(labels, cluster_labels)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

print("\nGenerating t-SNE visualization...\n")
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=5)
plt.colorbar(scatter, ticks=range(10))
plt.title("t-SNE Visualization of Latent Space")
plt.show(block=True)
input("Press Enter to close the t-SNE plot window...")

print("\nGenerating reconstruction visualization...\n")

def plot_images(original, reconstructed, n=6):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)
    input("Press Enter to close the reconstructed images plot window...")

test_images, _ = next(iter(train_loader))
test_images = test_images.to(device)

with torch.no_grad():
    reconstructed, _ = model(test_images)

plot_images(test_images.cpu(), reconstructed.cpu())
