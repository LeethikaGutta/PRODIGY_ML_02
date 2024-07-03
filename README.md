# PRODIGY_ML_02
Task
Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.
Dataset
https://www.kaggle.com/datasets/vjchoudharyZ/custome
r-segmentation-tutorial-in-python
code
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data: customer purchase history (example)
# Replace this with your actual dataset
# Example data:
# CustomerID, PurchaseCount, PurchaseAmount
# 1, 5, 250
# 2, 10, 600
# 3, 3, 100
# ...
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PurchaseCount': [5, 10, 3, 8, 12, 4, 6, 2, 9, 7],
    'PurchaseAmount': [250, 600, 100, 400, 700, 150, 300, 180, 500, 350]
}
df = pd.DataFrame(data)

# Selecting features for clustering (PurchaseCount and PurchaseAmount)
X = df[['PurchaseCount', 'PurchaseAmount']]

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying K-means clustering
k = 3  # Number of clusters (you can adjust this)
kmeans = KMeans(n_clusters=k, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing the clusters
plt.figure(figsize=(8, 6))

for cluster in range(k):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['PurchaseCount'], cluster_data['PurchaseAmount'], label=f'Cluster {cluster}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red', label='Centroids')
plt.title('Customer Segmentation based on Purchase Behavior')
plt.xlabel('Purchase Count')
plt.ylabel('Purchase Amount')
plt.legend()
plt.grid(True)
plt.show()

# Displaying the clusters
print(df[['CustomerID', 'Cluster']])
