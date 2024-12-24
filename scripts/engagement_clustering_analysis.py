import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from extract_data import connect_to_db  # Import the function to fetch data from DB
from clean_data import clean_data  # Import the function to clean data


def aggregate_user_engagement_metrics(df):
    """
    Aggregate user engagement metrics like session frequency, session duration,
    and session total traffic per customer (user_id).
    """
    # Aggregate data per user_id
    engagement_metrics = df.groupby('user_id').agg(
        session_frequency=('xdr_sessions', 'sum'),
        session_duration=('total_session_duration', 'sum'),
        total_traffic=('total_download_data', 'sum')
    ).reset_index()

    # Add upload data to total traffic
    engagement_metrics['total_traffic'] += df.groupby('user_id')['total_upload_data'].sum().values

    # Top 10 customers based on each engagement metric
    top_10_frequency = engagement_metrics.nlargest(10, 'session_frequency')
    top_10_duration = engagement_metrics.nlargest(10, 'session_duration')
    top_10_traffic = engagement_metrics.nlargest(10, 'total_traffic')

    return engagement_metrics, top_10_frequency, top_10_duration, top_10_traffic


def normalize_engagement_metrics(engagement_metrics):
    """
    Normalize engagement metrics using StandardScaler and return normalized data.
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(
        engagement_metrics[['session_frequency', 'session_duration', 'total_traffic']]
    )
    normalized_df = pd.DataFrame(normalized_data, columns=['session_frequency', 'session_duration', 'total_traffic'])
    return normalized_df


def k_means_clustering(normalized_df, k=3):
    """
    Apply K-means clustering to classify users into k engagement clusters.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    normalized_df['cluster'] = kmeans.fit_predict(normalized_df)

    # Calculate silhouette score to evaluate clustering performance
    silhouette_avg = silhouette_score(
        normalized_df[['session_frequency', 'session_duration', 'total_traffic']], normalized_df['cluster']
    )
    print(f"Silhouette Score for k={k}: {silhouette_avg}")
    return normalized_df, kmeans


def cluster_statistics(normalized_df, engagement_metrics):
    """
    Calculate statistics (min, max, average, total) of non-normalized metrics per cluster.
    """
    # Merge the non-normalized data with the cluster labels
    engagement_metrics['cluster'] = normalized_df['cluster']
    
    cluster_stats = engagement_metrics.groupby('cluster').agg(
        min_sessions=('session_frequency', 'min'),
        max_sessions=('session_frequency', 'max'),
        avg_sessions=('session_frequency', 'mean'),
        total_sessions=('session_frequency', 'sum'),
        min_duration=('session_duration', 'min'),
        max_duration=('session_duration', 'max'),
        avg_duration=('session_duration', 'mean'),
        total_duration=('session_duration', 'sum'),
        min_traffic=('total_traffic', 'min'),
        max_traffic=('total_traffic', 'max'),
        avg_traffic=('total_traffic', 'mean'),
        total_traffic=('total_traffic', 'sum')
    ).reset_index()

    return cluster_stats


def plot_engagement_clusters(engagement_metrics, cluster_stats):
    """
    Plot user engagement clusters based on K-means clustering.
    """
    # Plotting the engagement clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(
        engagement_metrics['session_frequency'],
        engagement_metrics['session_duration'],
        c=engagement_metrics['cluster'],
        cmap='viridis'
    )
    plt.xlabel('Session Frequency')
    plt.ylabel('Session Duration')
    plt.title('User Engagement Clusters')
    plt.colorbar(label='Cluster')
    plt.show()

    # Plotting the total traffic per cluster
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_stats['cluster'], cluster_stats['total_traffic'], color='skyblue')
    plt.xlabel('Cluster')
    plt.ylabel('Total Traffic (Bytes)')
    plt.title('Total Traffic per Engagement Cluster')
    plt.grid(True)
    plt.show()


def top_applications_per_engagement(df):
    """
    Aggregate user engagement metrics per application and identify the top 10 users per application.
    """
    app_engagement = df.groupby(['application', 'user_id']).agg(
        total_data_volume=('total_volume', 'sum'),
        total_session_duration=('total_session_duration', 'sum'),
        session_frequency=('xdr_sessions', 'sum')
    ).reset_index()

    top_app_users = app_engagement.groupby('application').apply(
        lambda x: x.nlargest(10, 'total_data_volume')
    ).reset_index(drop=True)

    return top_app_users


def plot_top_applications(app_engagement):
    """
    Plot the top 3 most used applications based on total data volume.
    """
    top_apps = app_engagement.groupby('application').agg(
        total_data_volume=('total_data_volume', 'sum')
    ).reset_index()

    top_apps = top_apps.nlargest(3, 'total_data_volume')

    plt.figure(figsize=(10, 6))
    plt.bar(top_apps['application'], top_apps['total_data_volume'], color='skyblue')
    plt.xlabel('Application')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.title('Top 3 Most Used Applications')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def elbow_method(df):
    """
    Elbow method to determine optimal number of clusters (k).
    """
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[['session_frequency', 'session_duration', 'total_traffic']])
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()


# Main execution
if __name__ == "__main__":
    # Fetch data from PostgreSQL
    df = connect_to_db()

    if df is not None:
        # Clean the data using the function from clean_data.py
        df = clean_data(df)

        # Aggregate user engagement metrics
        engagement_metrics, top_10_frequency, top_10_duration, top_10_traffic = aggregate_user_engagement_metrics(df)

        # Normalize the engagement metrics
        normalized_df = normalize_engagement_metrics(engagement_metrics)

        # Perform K-Means clustering with k=3
        normalized_df, kmeans = k_means_clustering(normalized_df, k=3)

        # Calculate statistics for each cluster
        cluster_stats = cluster_statistics(normalized_df, engagement_metrics)

        # Plot engagement clusters
        plot_engagement_clusters(normalized_df, cluster_stats)

        # Aggregate total traffic per application and find top 10 users
        top_app_users = top_applications_per_engagement(df)

        # Plot top 3 most used applications
        plot_top_applications(top_app_users)

        # Apply elbow method to find optimal k for clustering
        elbow_method(normalized_df)

        # Print results for top customers per engagement metric
        print("Top 10 Customers by Session Frequency:")
        print(top_10_frequency)

        print("Top 10 Customers by Session Duration:")
        print(top_10_duration)

        print("Top 10 Customers by Total Traffic:")
        print(top_10_traffic)

        # Print cluster stats
        print("Cluster Stats:")
        print(cluster_stats)
        
        print("Top Users per Application:")
        print(top_app_users)

    else:
        print("Failed to fetch data.")
