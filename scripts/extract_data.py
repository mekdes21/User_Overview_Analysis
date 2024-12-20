import psycopg2
import pandas as pd

def connect_to_db():
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
    finally:
        cursor.close()
        conn.close()

# Load data
df = connect_to_db()
print(df.head())
