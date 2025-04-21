import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load Data
data = load_iris()
X = data.data
y = data.target
labels = data.target_names

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Step 4: Scree Plot (Explained variance)
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# Step 5: Visualize 2D PCA projection
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(6,5))
for i in range(3):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=labels[i])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('2D PCA Projection')
plt.legend()
plt.grid(True)
plt.show()
