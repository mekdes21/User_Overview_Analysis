from matplotlib import ticker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from clean_data import clean_data
from aggregate_data import aggregate_user_data
from extract_data import connect_to_db

def describe_data(df):
    # Show summary statistics
    summary_stats = df.describe()
    print(summary_stats)
    return summary_stats

def perform_bivariate_analysis(df):
    # Check if the 'total_data' column exists and print column names
    if 'total_data' not in df.columns:
        print("Error: 'total_data' column not found in DataFrame.")
        print("Columns available: ", df.columns)
        return

    # Bivariate analysis: application vs total data
    plt.figure(figsize=(12, 6))  # Increase figure size for more space
    
    sns.boxplot(x='application', y='total_data', data=df)
    
    # Rotate x-axis labels and adjust their position
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust angle and alignment
    
    # Adjust the x-axis tick spacing manually using MaxNLocator
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='both', nbins=10))
    
    # Increase space between ticks
    plt.gca().tick_params(axis='x', which='major', pad=15)  # Increase the space between ticks
    
    # Adjust layout to provide more space for labels
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin
    
    plt.show()

def perform_correlation_analysis(df):
    # Check if necessary columns exist before performing correlation analysis
    required_columns = ['download_data', 'upload_data', 'total_data']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more columns: {required_columns}")
        return

    # Compute correlation matrix
    correlation_matrix = df[required_columns].corr()
    
    # Plot correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.show()

def perform_pca(df):
    # Check if necessary columns exist before performing PCA
    required_columns = ['session_duration', 'download_data', 'upload_data', 'total_data']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing one or more columns: {required_columns}")
        return

    # Standardize the data
    features = required_columns
    scaled_data = StandardScaler().fit_transform(df[features])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])

    # Plot PCA results
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Print explained variance ratio
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    return pca, pca_df

# Main function to perform EDA
def main():
    # Extract data from extract_data.py
    df = connect_to_db()  # Use the function from extract_data.py

    # Clean the data
    cleaned_df = clean_data(df)

    # Strip leading/trailing spaces from column names to avoid issues
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # Add total_data as the sum of download_data and upload_data
    cleaned_df['total_data'] = cleaned_df['download_data'] + cleaned_df['upload_data']

    # Print column names to ensure 'total_data' exists
    print("Columns in cleaned data:", cleaned_df.columns)

    # Describe data
    describe_data(cleaned_df)

    # Perform bivariate analysis
    perform_bivariate_analysis(cleaned_df)

    # Perform correlation analysis
    perform_correlation_analysis(cleaned_df)

    # Perform PCA
    pca, pca_df = perform_pca(cleaned_df)

    # Perform data aggregation (e.g., decile analysis)
    aggregated_data, total_data_per_decile = aggregate_user_data(cleaned_df)
    print(aggregated_data)
    print(total_data_per_decile)

if __name__ == "__main__":
    main()
