import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with a backend that works for you

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Load your dataset
file_path = '/Users/yagmursahin/Downloads/sample_data.csv' # Update this to your file's path
df = pd.read_csv(file_path)

# Assuming 'Age', 'Total Spent', and 'Purchase Frequency' are your columns of interest
features = df[['Age', 'Total Spent', 'Purchase Frequency']]

# Scale the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(scaled_features)
df['cluster'] = kmeans.labels_

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Total Spent'], df['Purchase Frequency'], c=df['cluster'], cmap='viridis')

# Adding labels and title
ax.set_title('3D Cluster Plot')
ax.set_xlabel('Age')
ax.set_ylabel('Total Spent')
ax.set_zlabel('Purchase Frequency')

# Legend
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

print(df.head(10))
# Show the plot
plt.show()
