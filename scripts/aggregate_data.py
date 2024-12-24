import pandas as pd
import matplotlib.pyplot as plt
from extract_data import connect_to_db  # Import the function from extract_data.py
from clean_data import clean_data  # Import the clean_data function from clean_data.py
import os

def aggregate_user_data(df):
    # Print the column names to inspect the data
    print("Columns in the DataFrame:", df.columns)

    # Rename columns to ensure they match expected names
    df = df.rename(columns={
        "Handset Manufacturer": "handset_manufacturer",
        "Handset Type": "handset_type"
    })

    # Step 1: Select relevant volume columns (e.g., social media, google, email, etc.)
    volume_columns = [
        'social_media_volume', 'google_volume', 'email_volume', 
        'youtube_volume', 'netflix_volume', 'gaming_volume', 'other_volume'
    ]

    # Step 2: Check if the volume columns exist in the dataframe
    if not all(col in df.columns for col in volume_columns):
        print("One or more volume columns are missing.")
        return None, None

    # Step 3: Drop the existing 'total_data_volume' column if it exists
    df = df.drop(columns=['total_data_volume'], errors='ignore')

    # Step 4: Melt the data to reshape it for aggregation (only for volume columns)
    df_melted = df.melt(id_vars=['user_id', 'handset_manufacturer', 'handset_type'],
                        value_vars=volume_columns,
                        var_name='application',
                        value_name='total_data_volume')

    # Step 5: Normalize the 'application' column values (remove 'volume' part)
    df_melted['application'] = df_melted['application'].str.replace('_volume', '').str.strip().str.lower()

    # Step 6: Now, merge the original DataFrame with the melted DataFrame to keep session information
    df_merged = pd.merge(df_melted, df[['user_id', 'total_session_duration', 'total_download_data', 'total_upload_data']], on='user_id', how='left')
    # DEBUG: Print the first few rows and columns of the merged DataFrame
    print("Columns in df_merged:", df_merged.columns)  # Print column names
    print(df_merged.head())  # Print first 5 rows of df_merged
    # Step 7: Group by user_id and application, and aggregate the metrics
    aggregated_data = df_merged.groupby(['user_id', 'application']).agg(
        number_of_sessions=('total_data_volume', 'count'),  # Count of records (sessions)
        total_data_volume=('total_data_volume', 'sum'),    # Sum of data volume
        total_session_duration=('total_session_duration', 'sum'),  # Total session duration
        total_download_data=('total_download_data', 'sum'),  # Total download data
        total_upload_data=('total_upload_data', 'sum')     # Total upload data
    ).reset_index()

    # Step 8: Handle missing session duration values by filling NaN with 0
    aggregated_data['total_session_duration'] = aggregated_data['total_session_duration'].fillna(0)
    aggregated_data['total_download_data'] = aggregated_data['total_download_data'].fillna(0)
    aggregated_data['total_upload_data'] = aggregated_data['total_upload_data'].fillna(0)

    # Step 9: Save the aggregated data to a CSV file (Check if output directory exists)
    output_dir = '../data/'  # Changed to local directory
    os.makedirs(output_dir, exist_ok=True)
    aggregated_data.to_csv(os.path.join(output_dir, 'aggregated_data_per_application.csv'), index=False)

    # Step 10: Visualization for total data per application
    output_dir_png = '../outputs/'  # Changed to local directory
    os.makedirs(output_dir_png, exist_ok=True)

    # Plot total data per application
    app_data_volume = aggregated_data.groupby('application')['total_data_volume'].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(app_data_volume['application'], app_data_volume['total_data_volume'], color='skyblue')
    plt.xlabel('Application')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.title('Total Data Volume per Application')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.savefig(os.path.join(output_dir_png, 'total_data_per_application.png'))  # Save plot to outputs folder
    plt.show()

    return aggregated_data, app_data_volume


def segment_users_into_deciles(df):
    # Compute total session duration and data volume for each user
    df['total_data_volume'] = df['total_download_data'] + df['total_upload_data']

    # Segment users into deciles based on total session duration
    df['duration_decile'] = pd.qcut(df['total_session_duration'], 10, labels=False)  # Deciles 0-9

    # Filter only the top 5 deciles (5–9)
    df_top_deciles = df[df['duration_decile'] >= 5]

    # Group data by decile and compute total data volume per decile
    decile_summary = df_top_deciles.groupby('duration_decile').agg(
        total_data_volume=('total_data_volume', 'sum'),
        avg_session_duration=('total_session_duration', 'mean'),
        total_session_duration=('total_session_duration', 'sum'),
        user_count=('user_id', 'nunique')
    ).reset_index()

    print("Decile Summary:")
    print(decile_summary)
    
    output_dir_png = '../outputs/'
    # Save decile summary to CSV
    output_dir = '../data/'
    os.makedirs(output_dir, exist_ok=True)
    decile_summary.to_csv(os.path.join(output_dir, 'decile_summary.csv'), index=False)

    # Visualization - Total Data Volume per Decile
    plt.figure(figsize=(10, 6))
    plt.bar(decile_summary['duration_decile'], decile_summary['total_data_volume'], color='skyblue')
    plt.xlabel('Decile Class')
    plt.ylabel('Total Data Volume (Bytes)')
    plt.title('Total Data Volume per Decile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_png, 'total_data_volume_deciles.png'))
    plt.show()

    return decile_summary


# Main execution
if __name__ == "__main__":
    # Fetch data from PostgreSQL using the function from extract_data.py
    df = connect_to_db()

    if df is not None:
        # Clean the data using the function from clean_data.py
        df = clean_data(df)
        
        # Perform Decile Segmentation and Analysis
        decile_summary = segment_users_into_deciles(df)

        # Aggregate data by application and user
        aggregated_data, app_data_volume = aggregate_user_data(df)

        if aggregated_data is not None:
            # Print the aggregated data to verify the result
            print("Aggregated Data:")
            print(aggregated_data.head())
            
            print("Decile Summary:")
            print(decile_summary)
            
            # Print the total data per application
            print("Total Data per Application:")
            print(app_data_volume)
        else:
            print("No relevant data to process.")
    else:
        print("Failed to fetch data.")
