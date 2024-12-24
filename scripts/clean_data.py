from extract_data import connect_to_db  # Import the function from extract_data.py

def clean_data(df):
    # Print the columns to help with debugging
    print("Columns in the DataFrame:", df.columns)
    
    # Check for required columns (excluding 'application' since it's generated later)
    required_columns = [
        'total_session_duration', 'total_download_data', 'total_upload_data', 
        'user_id', 'handset_type', 'social_media_volume', 'google_volume', 
        'email_volume', 'youtube_volume', 'netflix_volume', 'gaming_volume', 
        'other_volume'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Outlier handling using the IQR method
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Apply outlier removal to relevant columns
    for column in ['total_session_duration', 'total_download_data', 'total_upload_data', 
                   'social_media_volume', 'google_volume', 'email_volume', 
                   'youtube_volume', 'netflix_volume', 'gaming_volume', 'other_volume']:
        df = remove_outliers(df, column)

    # Handle missing values
    df['total_session_duration'] = df['total_session_duration'].fillna(df['total_session_duration'].mean())
    df['total_download_data'] = df['total_download_data'].fillna(df['total_download_data'].mean())
    df['total_upload_data'] = df['total_upload_data'].fillna(df['total_upload_data'].mean())
    df['social_media_volume'] = df['social_media_volume'].fillna(0)  # Assuming 0 for missing volumes
    df['google_volume'] = df['google_volume'].fillna(0)
    df['email_volume'] = df['email_volume'].fillna(0)
    df['youtube_volume'] = df['youtube_volume'].fillna(0)
    df['netflix_volume'] = df['netflix_volume'].fillna(0)
    df['gaming_volume'] = df['gaming_volume'].fillna(0)
    df['other_volume'] = df['other_volume'].fillna(0)
    df['user_id'] = df['user_id'].fillna(df['user_id'].mode()[0])
    df['handset_type'] = df['handset_type'].fillna('Unknown')  # Handle missing handset type

    # Create 'application' column by classifying based on volume columns
    def categorize_application(row):
        if row['social_media_volume'] > 0:
            return 'social_media'
        elif row['google_volume'] > 0:
            return 'google'
        elif row['email_volume'] > 0:
            return 'email'
        elif row['youtube_volume'] > 0:
            return 'youtube'
        elif row['netflix_volume'] > 0:
            return 'netflix'
        elif row['gaming_volume'] > 0:
            return 'gaming'
        else:
            return 'others'

    # Apply the categorization function to create the 'application' column
    df['application'] = df.apply(categorize_application, axis=1)

    # Calculate the total data volume after cleaning
    df['total_data_volume'] = df['total_download_data'] + df['total_upload_data']

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
