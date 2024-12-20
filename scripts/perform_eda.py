# Import the clean_data function from clean_data.py
from clean_data import clean_data
from extract_data import connect_to_db  # Import your data fetching function

# Function to describe the data (EDA)
def describe_data(df):
    # Show summary statistics
    summary_stats = df.describe()

    # Display the summary statistics for the cleaned dataset
    print(summary_stats)

    return summary_stats

# Fetch and clean the data
df = connect_to_db()  # Fetch data from the database
cleaned_df = clean_data(df)  # Clean the data

# Perform EDA and display the summary statistics
eda_summary = describe_data(cleaned_df)
