import os
from extract_data import connect_to_db  # Import the function from extract_data.py

def aggregate_user_data(df):
    # Group by user_id and application, and aggregate the metrics
    aggregated_data = df.groupby('user_id').agg(
        number_of_sessions=('session_duration', 'count'),
        total_session_duration=('session_duration', 'sum'),
        total_download_data=('download_data', 'sum'),
        total_upload_data=('upload_data', 'sum')
    ).reset_index()

    # Add the total data volume by summing download and upload data
    aggregated_data['total_data_volume'] = aggregated_data['total_download_data'] + aggregated_data['total_upload_data']

    # Save the aggregated data to a CSV file
    aggregated_data.to_csv('../data/aggregated_data.csv', index=False)
    return aggregated_data

# Fetch data from PostgreSQL using the function from extract_data.py
df = connect_to_db()

# Aggregate data
aggregated_data = aggregate_user_data(df)
print(aggregated_data.head())  # Print the aggregated data to verify the result
