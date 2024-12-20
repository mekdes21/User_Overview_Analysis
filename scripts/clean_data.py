from extract_data import connect_to_db  # Import the function from extract_data.py

def clean_data(df):
    # Fill missing values with mean for numeric columns
    df['session_duration'] = df['session_duration'].fillna(df['session_duration'].mean())
    df['download_data'] = df['download_data'].fillna(df['download_data'].mean())
    df['upload_data'] = df['upload_data'].fillna(df['upload_data'].mean())

    return df

# Fetch data from PostgreSQL using the function from extract_data.py
df = connect_to_db()

# Clean the data
cleaned_df = clean_data(df)

# Optionally, you can print the cleaned data to verify
print(cleaned_df.head())
