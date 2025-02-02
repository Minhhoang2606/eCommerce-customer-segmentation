'''
eCommerce customer segmentation
Author: Henry Ha
'''
# Importing all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Load the dataset
df = pd.read_csv('cust_data.csv')

# Display dataset information
print(df.info())

# Display the first few rows
print(df.head())

# Display basic statistics
print(df.describe())

# Fill missing values in 'Gender' with the mode
gender_mode = df['Gender'].mode()[0]
df['Gender'].fillna(gender_mode, inplace=True)

# Verify missing values
print(df['Gender'].isnull().sum())

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Plot gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender')

# Add the value of each bar on top
for p in plt.gca().patches:
    plt.gca().text(p.get_x() + p.get_width() / 2, p.get_height() + 0.5, f'{p.get_height():.0f}', ha='center', va='bottom')

plt.title('Gender Distribution')
plt.show()

# Plot overall order count and gender breakdown
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Orders')
plt.title('Overall Orders')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Orders', hue='Gender')
plt.title('Orders by Gender')
plt.show()

# Create a new dataframe
new_df = df.copy()

# Create the 'Total Search' column
new_df['Total Search'] = new_df.iloc[:, 3:].sum(axis=1)

# Display the first few rows
print(new_df.head())

# Select top brands based on total interactions
top_brands = new_df.iloc[:, 3:-1].sum().sort_values(ascending=False).head(10)

# Plot bar chart for top brand interactions
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=top_brands.index, y=top_brands.values)
plt.title('Top 10 Brands by Interaction Frequency')
plt.ylabel('Total Interactions')
plt.xlabel('Brand')
plt.xticks(rotation=45)

# Add the value of each bar on top
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='center',
                xytext=(0, 10),
                textcoords='offset points')

plt.show()

# Top 10 customers based on Total Search
plt_data = new_df.sort_values('Total Search', ascending=False)[['Cust_ID', 'Gender', 'Total Search']].head(10)

# Plot
sns.barplot(data=plt_data,
            x='Cust_ID',
            y='Total Search',
            hue='Gender',
            order=plt_data.sort_values('Total Search', ascending=False).Cust_ID)
plt.title("Top 10 Customers Based on Total Searches")
plt.show()

# Plot histograms for all features from 'Orders' onwards
new_df.iloc[:, 2:].hist(figsize=(40, 30))
plt.show()

# Generate boxplots for orders and brand-specific columns
cols = list(new_df.columns[2:])
def dist_list(lst):
    plt.figure(figsize=(40, 30))
    for i, col in enumerate(lst, 1):
        plt.subplot(6, 6, i)
        sns.boxplot(data=new_df, x=new_df[col])
        plt.title(col)
    plt.tight_layout()

dist_list(cols)

# Correlation heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(df.iloc[:, 2:].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pair plot for selected features
sns.pairplot(new_df[['Orders', 'Total Search', 'Samsung', 'Gatorade']])
plt.show()

#TODO Data preprocessing

# One-hot encoding the 'Gender' column
gender_encoded = pd.get_dummies(new_df['Gender'], prefix='Gender', drop_first=True)
new_df = pd.concat([new_df, gender_encoded], axis=1)

# Drop the original 'Gender' column
new_df.drop(columns=['Gender'], inplace=True)
print(new_df.head())

# Feature scaling
# Initialize the scaler
scaler = MinMaxScaler()

# Select numerical columns for scaling
numerical_columns = new_df.iloc[:, 2:].columns

# Scale the numerical features
new_df[numerical_columns] = scaler.fit_transform(new_df[numerical_columns])

print(new_df.head())

#TODO Clustering for Customer Segmentation

# Using KElbowVisualizer
kmeans = KMeans(random_state=42)
visualizer = KElbowVisualizer(kmeans, k=(1, 15), timings=False)
visualizer.fit(new_df.iloc[:, 2:])
visualizer.show()

# Calculate silhouette scores for different K values
silhouette_avg = []
for k in range(2, 16):  # Silhouette score requires at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(new_df.iloc[:, 2:])
    silhouette_avg.append(silhouette_score(new_df.iloc[:, 2:], cluster_labels))

# Plot Silhouette Scores
plt.figure(figsize=(10, 7))
plt.plot(range(2, 16), silhouette_avg, 'bX-')
plt.title('Silhouette Analysis for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Initialize K-Means model
kmeans = KMeans(random_state=42)

# Visualize the elbow curve
visualizer = KElbowVisualizer(kmeans, k=(1, 15), timings=False)
visualizer.fit(new_df.iloc[:, 2:])  # Exclude non-clustering columns like 'Cust_ID'
visualizer.show()

# Apply K-Means with the optimal number of clusters
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
new_df['Cluster'] = kmeans.fit_predict(new_df.iloc[:, 2:])
print(new_df.head())

# Scatter plot for clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=new_df, x='Total Search', y='Orders', hue='Cluster', palette='viridis')
plt.title('Customer Clusters Based on Total Search and Orders')
plt.xlabel('Total Search')
plt.ylabel('Orders')
plt.legend(title='Cluster')
plt.show()


# Silhouette Visualization
# Exclude 'Cluster' column to match features during fitting
features_for_visualization = new_df.drop(columns=['Cluster'])

# Apply K-Means for K = 4
kmeans_k4 = KMeans(n_clusters=4, random_state=42)

# Silhouette visualization for K = 4
visualizer = SilhouetteVisualizer(kmeans_k4, colors='yellowbrick')
visualizer.fit(features_for_visualization.iloc[:, 2:])  # Exclude non-relevant columns
visualizer.show()

from scipy.cluster.hierarchy import dendrogram, linkage

# Perform hierarchical clustering
linked = linkage(new_df.iloc[:, 2:], method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.show()

#TODO Visualizing and Evaluating Clusters

# Visualize cluster size distribution
sns.countplot(x='Cluster', data=new_df, palette=['purple', 'blue', 'green', 'yellow'])
plt.title('Cluster Size Distribution')
plt.ylabel('Count')
plt.xlabel('Cluster')
plt.show()

# Customer count by gender within each cluster
sns.countplot(x='Gender', hue='Cluster', data=new_df, palette=['purple', 'blue', 'green', 'yellow'])
plt.title('Customer Count by Gender in Each Cluster')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.show()

# Total searches by gender within each cluster
gender_searches = new_df.groupby(['Cluster', 'Gender'])['Total Search'].sum().reset_index()
sns.barplot(x='Cluster', y='Total Search', hue='Gender', data=gender_searches)
plt.title('Total Searches by Gender in Each Cluster')
plt.ylabel('Total Searches')
plt.xlabel('Cluster')
plt.show()

# Total orders made by each cluster
cluster_orders = new_df.groupby('Cluster')['Orders'].sum().reset_index()
sns.barplot(x='Cluster', y='Orders', data=cluster_orders, palette=['purple', 'blue', 'green', 'yellow'])
plt.title('Past Orders by Cluster')
plt.ylabel('Orders')
plt.xlabel('Cluster')
plt.show()

# Identify the top 10 products based on overall interactions
top_10_products = new_df.iloc[:, 3:].sum(axis=0).sort_values(ascending=False).head(10).index.tolist()

# Add the top 10 products to the relevant features
heatmap_features = ['Total Search', 'Orders'] + top_10_products

# Group by clusters and calculate the mean for the selected features
heatmap_data = new_df.groupby('Cluster')[heatmap_features].mean()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5)
plt.title("Cluster Feature Analysis Heatmap (with Top 10 Products)")
plt.xlabel("Features")
plt.ylabel("Clusters")
plt.show()

