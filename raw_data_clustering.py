from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

# Step 1: Load MNIST data as flat vectors
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

data = []
labels = []

# Step 2: Convert each image to a flattened numpy array
for img, label in mnist_train:
    data.append(img.numpy().flatten())
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

print(f"Data shape (num_samples, features): {data.shape}")
print(f"Labels shape: {labels.shape}")

# Step 3: Run KMeans clustering on raw data
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(data)

# Step 4: Calculate clustering evaluation metrics
sil_score = silhouette_score(data, cluster_labels)
db_score = davies_bouldin_score(data, cluster_labels)

print(f"Silhouette Score (raw data): {sil_score:.4f}")
print(f"Davies-Bouldin Index (raw data): {db_score:.4f}")
