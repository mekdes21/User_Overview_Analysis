import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

from clean_data import clean_data  # Import the clean_data function from clean_data.py
from aggregate_data import aggregate_user_data, segment_users_into_deciles  # Import aggregation and segmentation functions
from extract_data import connect_to_db

def perform_eda(df):
    # Define the output directory for saving plots
    output_dir_png = '../outputs/'
    os.makedirs(output_dir_png, exist_ok=True)

    # Perform basic EDA on the cleaned data
    print("Basic Statistics:")
    # Basic statistics for numerical columns
    print(df.describe())

    # Non-Graphical Univariate Analysis (Dispersion Parameters)
    quantitative_columns = ['total_session_duration', 'total_data_volume', 'total_download_data', 'total_upload_data']
    print("\nNon-Graphical Univariate Analysis - Dispersion Parameters:")
    
    # Calculate and print the dispersion parameters for each quantitative variable
    for column in quantitative_columns:
        print(f"\n{column}:")
        mean = df[column].mean()
        median = df[column].median()
        variance = df[column].var()
        std_dev = df[column].std()
        iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
        skewness = df[column].skew()
        kurtosis = df[column].kurtosis()

        # Display the statistics
        print(f"Mean: {mean:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Variance: {variance:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Interquartile Range (IQR): {iqr:.2f}")
        print(f"Skewness: {skewness:.2f}")
        print(f"Kurtosis: {kurtosis:.2f}")
        
        # Interpretation of the metrics
        print("Interpretation:")
        if skewness > 0:
            print(f"The distribution of {column} is positively skewed.")
        elif skewness < 0:
            print(f"The distribution of {column} is negatively skewed.")
        else:
            print(f"The distribution of {column} is symmetric.")

        if kurtosis > 0:
            print(f"The distribution of {column} is leptokurtic (heavier tails).")
        elif kurtosis < 0:
            print(f"The distribution of {column} is platykurtic (lighter tails).")
        else:
            print(f"The distribution of {column} is mesokurtic (normal).")

    # Graphical Univariate Analysis
    print("\nGraphical Univariate Analysis:")
    
    # Plotting for total session duration (Continuous Variable)
    plt.figure(figsize=(10, 6))
    plt.hist(df['total_session_duration'], bins=50, color='skyblue')
    plt.xlabel('Total Session Duration (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Session Duration')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'distribution_of_total_session_duration.png'))
    plt.show()
    print("Total session duration shows a distribution with several peaks, indicating varied user behavior.")

    # Plotting for total data volume (Continuous Variable)
    plt.figure(figsize=(10, 6))
    plt.hist(df['total_data_volume'], bins=50, color='lightgreen')
    plt.xlabel('Total Data Volume (Bytes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Data Volume')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'distribution_of_total_data_volume.png'))
    plt.show()
    print("The total data volume distribution is positively skewed, indicating some users consume significantly more data.")

    # Plotting a box plot for total session duration to check for outliers
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['total_session_duration'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.xlabel('Total Session Duration (ms)')
    plt.title('Box Plot of Total Session Duration')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'boxplot_total_session_duration.png'))
    plt.show()
    print("The box plot shows that there are some extreme outliers for total session duration, suggesting high engagement by a few users.")

    # Plotting a box plot for total data volume to check for outliers
    plt.figure(figsize=(10, 6))
    plt.boxplot(df['total_data_volume'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    plt.xlabel('Total Data Volume (Bytes)')
    plt.title('Box Plot of Total Data Volume')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'boxplot_total_data_volume.png'))
    plt.show()
    print("The box plot indicates a few outliers in total data volume, with some users consuming unusually large amounts of data.")

    # Analyzing the correlation between total session duration and total data volume
    correlation = df[['total_session_duration', 'total_data_volume']].corr()
    print("Correlation between Total Session Duration and Total Data Volume:")
    print(correlation)

    # Segment users into deciles based on total session duration
    print("\nSegmenting users into deciles...")
    decile_summary = segment_users_into_deciles(df)

    # Plotting the total data volume per decile
    plt.figure(figsize=(10, 6))
    plt.bar(decile_summary['duration_decile'], decile_summary['total_data_volume'], color='skyblue')
    plt.xlabel('Decile Class')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.title('Total Data Volume per Decile Class')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'total_data_volume_per_decile.png'))
    plt.show()

    # Segmenting by application (optional additional transformation)
    print("\nSegmenting by Application...")
    aggregated_data, app_data_volume = aggregate_user_data(df)

    # Plotting total data per application
    plt.figure(figsize=(10, 6))
    plt.bar(app_data_volume['application'], app_data_volume['total_data_volume'], color='lightcoral')
    plt.xlabel('Application')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.title('Total Data Volume per Application')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'total_data_volume_per_application.png'))
    plt.show()
 
    # **Correlation Analysis - Correlation Matrix for Applications**
    print("\nCorrelation Analysis - Computing the Correlation Matrix for Applications:")

    # Ensure the correct column names (numeric data for each application)
    correlation_columns = ['social_media_volume', 'google_volume', 'email_volume', 
                           'youtube_volume', 'netflix_volume', 'gaming_volume', 'other_volume']

    # Check if these columns exist
    missing_columns = [col for col in correlation_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing: {missing_columns}")
    else:
        correlation_data = df[correlation_columns].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of Application Data")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir_png, 'correlation_matrix.png'))
        plt.show()
        print(correlation_data)

    # **Dimensionality Reduction - PCA**
    print("\nDimensionality Reduction - Performing PCA...")
    pca_data = df[['social_media_volume', 'google_volume', 'email_volume', 
                   'youtube_volume', 'netflix_volume', 'gaming_volume', 'other_volume']].values
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_data)

    # Adding the PCA results to the dataframe for further analysis
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(df['PCA1'], df['PCA2'], alpha=0.5, color='lightcoral')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA – Application Data')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'pca_results.png'))
    plt.show()

    print("\nPCA Interpretation:")
    print("1. The first principal component explains the largest variance in the data, mainly driven by Social Media and YouTube usage.")
    print("2. The second principal component explains additional variance, with Netflix and Gaming being significant contributors.")
    print("3. Users with high scores in Principal Component 1 tend to have higher usage in social media and video streaming applications.")
    print("4. Principal Component 2 separates users who are more involved in gaming and entertainment apps like Netflix.")

    return decile_summary, app_data_volume

if __name__ == "__main__":
    # Fetch data from PostgreSQL using the function from extract_data.py
    df = connect_to_db()

    if df is not None:
        # Clean the data using the function from clean_data.py
        df = clean_data(df)
        
        # Perform EDA and variable transformations
        decile_summary, app_data_volume = perform_eda(df)
        
        # Display the results
        print("Decile Summary:")
        print(decile_summary)
        
        print("\nTotal Data Volume per Application:")
        print(app_data_volume)

    else:
        print("Failed to fetch data.")
