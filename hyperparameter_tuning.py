import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

latent_dim = 128
learning_rate = 0.001
num_epochs = 30
batch_size = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

input_size = 28 * 28
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
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

model = Autoencoder(latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []

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
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

plt.figure()
plt.plot(range(1, num_epochs+1), losses, marker='o')
plt.title(f'Training Loss Curve (latent_dim={latent_dim}, lr={learning_rate})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f'loss_curve_ld{latent_dim}_lr{learning_rate}.png')
plt.close()
print(f"Saved loss curve plot: loss_curve_ld{latent_dim}_lr{learning_rate}.png")

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

kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(latent_vectors)

sil_score = silhouette_score(latent_vectors, cluster_labels)
db_score = davies_bouldin_score(latent_vectors, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")
