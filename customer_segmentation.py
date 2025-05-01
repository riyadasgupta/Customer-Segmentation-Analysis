import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading Dataset
data = pd.read_csv("customer_segmentation_data.csv")

print("First 5 rows:", data.head())

print("\nMissing values:", data.isnull().sum())


# Selecting features for clustering
# using only simple numerical features 
features = data[['income', 'spending_score']]

# Standardizing the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Using the Elbow Method to find the best number of clusters
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_features)
    inertia.append(model.inertia_)

# Plotting the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Applying K-Means with 3 clusters 
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Result
print("\nClustered data:")
print(data[['income', 'spending_score', 'Cluster']].head())

# Visualizing the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['income'], data['spending_score'], c=data['Cluster'], cmap='viridis', s=50)
plt.title('Customer Segments')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.grid(True)
plt.show()

