import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load integrated dataset
df = pd.read_csv("data/processed/customers_transactions_integrated.csv")

# Aggregate by customer
customer_stats = df.groupby('customer_id').agg({
    'total_amount': 'sum',  # total spending
    'transaction_id': 'count',  # number of transactions (frequency)
}).reset_index()

customer_stats.columns = ['customer_id', 'total_spent', 'transaction_count']

# Calculate average transaction value
customer_stats['avg_transaction_value'] = customer_stats['total_spent'] / customer_stats['transaction_count']

print("Customer Statistics:")
print(customer_stats.head())
print(f"\nTotal customers: {len(customer_stats)}")

# Select features for clustering
features = ['total_spent', 'avg_transaction_value', 'transaction_count']
X = customer_stats[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
customer_stats['cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered dataset
customer_stats.to_csv("data/processed/clustered_dataset.csv", index=False)

# Results
print("\n=== CLUSTER RESULTS ===")
print(f"\nNumber of clusters: 3")

print("\nCluster distribution:")
cluster_counts = customer_stats['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    print(f"  Cluster {cluster_id}: {count} customers")

print("\nCluster Analysis:")
for cluster_id in sorted(customer_stats['cluster'].unique()):
    cluster_data = customer_stats[customer_stats['cluster'] == cluster_id]
    total_spending = cluster_data['total_spent'].sum()
    num_customers = len(cluster_data)
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Number of customers: {num_customers}")
    print(f"  Total spending: ${total_spending:.2f}")
    print(f"  Avg customer spending: ${cluster_data['total_spent'].mean():.2f}")
    print(f"  Avg transaction count: {cluster_data['transaction_count'].mean():.2f}")

# Find cluster with highest spending
cluster_spending = customer_stats.groupby('cluster')['total_spent'].sum()
highest_spending_cluster = cluster_spending.idxmax()
print(f"\n\n*** Cluster with highest total spending: Cluster {highest_spending_cluster} (${cluster_spending[highest_spending_cluster]:.2f}) ***")

# Find cluster with most frequent customers
cluster_frequency = customer_stats.groupby('cluster')['transaction_count'].mean()
most_frequent_cluster = cluster_frequency.idxmax()
print(f"*** Cluster with most frequent customers: Cluster {most_frequent_cluster} (avg {cluster_frequency[most_frequent_cluster]:.2f} transactions) ***")
