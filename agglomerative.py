# This program applies Agglomerative Hierarchical Clustering (a bottom-up hierarchical clustering method) on the Iris dataset to form 3 clusters corresponding to the three flower species — Setosa, Versicolor, and Virginica.
# Then, it compares the predicted clusters with actual labels to measure accuracy and visualize the confusion matrix and cluster plot.

# Agglomerative Hierarchical Clustering
#
# 1. What is Hierarchical Clustering?
# Unsupervised learning algorithm
# Builds a hierarchy of clusters
# Does not require labels
# Produces a tree-like structure (dendrogram)
#
# How Agglomerative Clustering Works
# Treat each point as a cluster
# Compute distance matrix
# Merge closest clusters using linkage rule
# Update distance matrix
# Repeat until n_clusters reac

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode

iris = load_iris()
X = iris.data
y_true = iris.target

# Creates an Agglomerative Clustering model with 3 clusters. linkage='ward' specifies Ward's method, which minimizes the within-cluster variance when merging clusters (tends to create compact, spherical clusters).
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred = agg.fit_predict(X)


labels = np.zeros_like(y_pred)
for i in range(3):
    mask = (y_pred == i)
    labels[mask] = mode(y_true[mask], keepdims=True).mode[0]


acc = accuracy_score(y_true, labels)
print(f"\nAgglomerative Clustering Accuracy: {acc:.4f}")
cm = confusion_matrix(y_true, labels)
print("\nConfusion Matrix:", cm)

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names).plot(cmap='Blues')
plt.title("Confusion Matrix - Agglomerative Clustering")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("Agglomerative Clustering Results")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()
