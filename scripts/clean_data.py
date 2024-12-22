from extract_data import connect_to_db  # Import the function from extract_data.py

def clean_data(df):
    # Check for required columns after renaming
    required_columns = ['session_duration', 'download_data', 'upload_data', 'application', 'user_id', 'handset_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Outlier handling using the IQR method
    def remove_outliers(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Apply outlier removal to relevant columns
    for column in ['session_duration', 'download_data', 'upload_data']:
        df = remove_outliers(column)

    # Handle missing values
    df['session_duration'] = df['session_duration'].fillna(df['session_duration'].mean())
    df['download_data'] = df['download_data'].fillna(df['download_data'].mean())
    df['upload_data'] = df['upload_data'].fillna(df['upload_data'].mean())
    df['application'] = df['application'].fillna('Unknown')
    df['user_id'] = df['user_id'].fillna(df['user_id'].mode()[0])
    df['handset_type'] = df['handset_type'].fillna('Unknown')  # Handle missing handset type

    # Calculate the total data volume after cleaning
    df['total_data_volume'] = df['download_data'] + df['upload_data']

    return df

# Fetch data from PostgreSQL using the function from extract_data.py
df = connect_to_db()

# Check if the fetched data is valid
if df is None or df.empty:
    print("No data fetched from the database.")
else:
    # Clean the data
    cleaned_df = clean_data(df)

    # Optionally, print the cleaned data to verify
    print("Cleaned Data:")
    print(cleaned_df.head())
