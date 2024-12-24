import psycopg2
import pandas as pd

# Function to connect to the database and fetch data
def connect_to_db():
    conn = None
    cursor = None
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            dbname="xdr_data_db", 
            user="postgres", 
            password="1111", 
            host="localhost", 
            port="5432"
        )

        # Create a cursor object
        cursor = conn.cursor()

        # Query to fetch the required data
        query = """
        SELECT 
            "MSISDN/Number" AS user_id,
            "Handset Manufacturer" AS handset_manufacturer,
            "Handset Type" AS handset_type,

            COUNT(*) AS xdr_sessions, -- Count sessions per user
            SUM("Dur. (ms)") AS total_session_duration, -- Total session duration
            SUM("Total DL (Bytes)") AS total_download_data, -- Total download data
            SUM("Total UL (Bytes)") AS total_upload_data, -- Total upload data

            -- Application-specific data
            SUM("Social Media DL (Bytes)") + SUM("Social Media UL (Bytes)") AS social_media_volume,
            SUM("Google DL (Bytes)") + SUM("Google UL (Bytes)") AS google_volume,
            SUM("Email DL (Bytes)") + SUM("Email UL (Bytes)") AS email_volume,
            SUM("Youtube DL (Bytes)") + SUM("Youtube UL (Bytes)") AS youtube_volume,
            SUM("Netflix DL (Bytes)") + SUM("Netflix UL (Bytes)") AS netflix_volume,
            SUM("Gaming DL (Bytes)") + SUM("Gaming UL (Bytes)") AS gaming_volume,
            SUM("Other DL (Bytes)") + SUM("Other UL (Bytes)") AS other_volume,

            -- Total volume combining all data
            SUM("Total DL (Bytes)") + SUM("Total UL (Bytes)") AS total_volume

        FROM 
            xdr_data
        GROUP BY 
            "MSISDN/Number", "Handset Manufacturer", "Handset Type" -- Group by user, manufacturer, and handset type
        ORDER BY 
            user_id;
        """
        cursor.execute(query)

        # Fetch the result into a DataFrame
        columns = ['user_id', 'handset_manufacturer', 'handset_type', 'xdr_sessions', 'total_session_duration', 'total_download_data', 'total_upload_data', 'social_media_volume', 'google_volume', 'email_volume', 'youtube_volume', 'netflix_volume', 'gaming_volume', 'other_volume', 'total_volume']
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=columns)

        return df
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Load data
if __name__ == "__main__":
    df = connect_to_db()

    if df is not None:
        # Display the columns of the DataFrame
        print("Columns in the fetched data:", df.columns)

        # Display the first few rows of the data
        print(df.head())
    else:
        print("Failed to fetch data.")
