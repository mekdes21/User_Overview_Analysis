import psycopg2
import pandas as pd

def connect_to_db():
    conn = None
    cursor = None
    try:
        # Connect to your postgres DB
        conn = psycopg2.connect(
            dbname="xdr_data_db", 
            user="postgres", 
            password="1111", 
            host="localhost", 
            port="5432"
        )

        # Create a cursor object
        cursor = conn.cursor()

        # Query to fetch the necessary data
        query = """
        SELECT "MSISDN/Number" AS user_id, 
               "Handset Manufacturer" AS application, 
               "Dur. (ms)" AS session_duration, 
               "Total DL (Bytes)" AS download_data, 
               "Total UL (Bytes)" AS upload_data
        FROM xdr_data;
        """
        cursor.execute(query)

        # Fetch the result into a DataFrame
        data = cursor.fetchall()
        columns = ['user_id', 'application', 'session_duration', 'download_data', 'upload_data']
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
df = connect_to_db()

if df is not None:
    # Check the columns of the DataFrame
    print("Columns in the fetched data:", df.columns)
    
    # Check the first few rows of the DataFrame to verify the data
    print(df.head())  
else:
    print("Failed to fetch data.")
