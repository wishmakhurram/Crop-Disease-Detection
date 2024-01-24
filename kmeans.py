import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, max_iters=100):

    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for _ in range(max_iters):

        labels = np.argmin(np.linalg.norm(X - centroids[:, np.newaxis], axis=2), axis=0)

        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    return centroids, labels


np.random.seed(42)
data = np.random.rand(100, 2) * 10


k = 3


centroids, labels = kmeans(data, k)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')  
#plt.legend()
plt.show()
