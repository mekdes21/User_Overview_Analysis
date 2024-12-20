import pandas as pd
from extract_data import connect_to_db  # Import the function from extract_data.py
from clean_data import clean_data  # Import the clean_data function from clean_data.py

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
    
    # Segment users into decile classes based on total session duration
    aggregated_data['decile'] = pd.qcut(aggregated_data['total_session_duration'], 10, labels=False)

    # Compute the total data (download + upload) per decile class
    total_data_per_decile = aggregated_data.groupby('decile')['total_data_volume'].sum().reset_index()
    
    # Add the aggregated data per decile to the original dataframe
    aggregated_data = aggregated_data.merge(total_data_per_decile, on='decile', suffixes=('', '_per_decile'))

    # Save the aggregated data to a CSV file
    aggregated_data.to_csv('../data/aggregated_data.csv', index=False)
    
    return aggregated_data, total_data_per_decile

# Fetch data from PostgreSQL using the function from extract_data.py
df = connect_to_db()

# Clean the data using the function from clean_data.py
df = clean_data(df)

# Aggregate data
aggregated_data, total_data_per_decile = aggregate_user_data(df)

# Print the aggregated data to verify the result
print("Aggregated Data:")
print(aggregated_data.head())

# Print the total data per decile
print("Total Data per Decile:")
print(total_data_per_decile)
