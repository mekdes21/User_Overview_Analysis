# experience_analytics.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from extract_data import connect_to_db
import os

# Create output directory if it doesn't exist
output_dir = '../outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
df = connect_to_db()

if df is not None:
    # Task 3.1 - Aggregating per customer information (Handling missing values)
    def aggregate_per_customer(df):
        # Replace missing values with mean (for numerical columns) or mode (for categorical)
        df['tcp_dl_retransmission'].fillna(df['tcp_dl_retransmission'].mean(), inplace=True)
        df['tcp_ul_retransmission'].fillna(df['tcp_ul_retransmission'].mean(), inplace=True)
        df['avg_rtt_dl'].fillna(df['avg_rtt_dl'].mean(), inplace=True)
        df['avg_rtt_ul'].fillna(df['avg_rtt_ul'].mean(), inplace=True)
        df['handset_type'].fillna(df['handset_type'].mode()[0], inplace=True)

        # Aggregate the required information
        aggregated_data = df.groupby(['user_id', 'handset_type']).agg({
            'tcp_dl_retransmission': 'mean',
            'tcp_ul_retransmission': 'mean',
            'avg_rtt_dl': 'mean',
            'avg_rtt_ul': 'mean',
            'total_download_data': 'mean',
            'total_upload_data': 'mean'
        }).reset_index()
        
        return aggregated_data

    aggregated_df = aggregate_per_customer(df)
    print(aggregated_df.head())

    # Task 3.2 - Compute & list 10 of the top, bottom, and most frequent:
    def list_top_bottom_frequent(df):
        top_tcp = df['tcp_dl_retransmission'].nlargest(10)
        bottom_tcp = df['tcp_dl_retransmission'].nsmallest(10)
        top_rtt = df['avg_rtt_dl'].nlargest(10)
        bottom_rtt = df['avg_rtt_dl'].nsmallest(10)
        top_throughput = df['total_download_data'].nlargest(10)
        bottom_throughput = df['total_download_data'].nsmallest(10)
        
        # Most frequent values
        most_frequent_tcp = df['tcp_dl_retransmission'].mode().head(10)
        most_frequent_rtt = df['avg_rtt_dl'].mode().head(10)
        most_frequent_throughput = df['total_download_data'].mode().head(10)
        
        return top_tcp, bottom_tcp, most_frequent_tcp, top_rtt, bottom_rtt, most_frequent_rtt, top_throughput, bottom_throughput, most_frequent_throughput

    top_tcp, bottom_tcp, most_frequent_tcp, top_rtt, bottom_rtt, most_frequent_rtt, top_throughput, bottom_throughput, most_frequent_throughput = list_top_bottom_frequent(df)
    print("Top TCP:", top_tcp)
    print("Bottom TCP:", bottom_tcp)

    # Task 3.3 - Distribution of average throughput per handset type
    def distribution_throughput(df):
        throughput_by_handset = df.groupby('handset_type')['total_download_data'].mean()

        # Limit the handset types to top 10 for better readability
        throughput_by_handset = throughput_by_handset.nlargest(10)

        # Shorten handset names if they are too long
        throughput_by_handset.index = throughput_by_handset.index.str[:10]  # Shorten names to the first 10 characters

        # Plotting the bar chart with adjusted figure size
        plt.figure(figsize=(18, 12))  # Set figure size for better spacing
        ax = throughput_by_handset.plot(kind='bar', title="Average Throughput per Handset Type")
        
        # Rotate labels 45 degrees for better space
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=12, color='black')
        
        # Adjust the bottom margin to prevent overlap with labels
        plt.subplots_adjust(bottom=0.3)  # Increased bottom margin
        
        # Customize label appearance
        ax.set_xlabel("Handset Type", labelpad=15, fontsize=14, color='darkblue', fontweight='bold')
        ax.set_ylabel("Average Throughput (MB)", fontsize=14, color='darkblue', fontweight='bold')

        # Adjust layout to avoid overlapping and ensure everything fits
        plt.tight_layout(pad=2.0)  # Ensure adequate space around plot elements

        # Save the plot
        plt.savefig(f"{output_dir}/throughput_per_handset_type.png")

        # Show the plot
        plt.show()

    distribution_throughput(df)

    # Task 3.4 - K-Means Clustering
    def kmeans_clustering(df):
        features = ['tcp_dl_retransmission', 'avg_rtt_dl', 'total_download_data']
        df_kmeans = df[features].dropna()
        
        kmeans = KMeans(n_clusters=3)
        clusters = kmeans.fit_predict(df_kmeans)
        df_kmeans['cluster'] = clusters
        
        # Plot the clusters with adjusted figure size
        plt.figure(figsize=(18, 10))  # Set figure size for better spacing
        plt.scatter(df_kmeans['tcp_dl_retransmission'], df_kmeans['avg_rtt_dl'], c=df_kmeans['cluster'])
        plt.xlabel("TCP Retransmission", fontsize=14, color='darkblue', fontweight='bold')
        plt.ylabel("Average RTT", fontsize=14, color='darkblue', fontweight='bold')
        plt.title("K-Means Clustering of User Experiences", fontsize=16, color='darkblue', fontweight='bold')

        # Save the plot
        plt.savefig(f"{output_dir}/kmeans_clustering.png")

        # Show the plot
        plt.show()

        return df_kmeans

    clustered_df = kmeans_clustering(df)
    print(clustered_df.head())
else:
    print("Failed to fetch data.")
