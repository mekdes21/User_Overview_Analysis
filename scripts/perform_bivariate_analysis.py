import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from clean_data import clean_data  # Assuming you have the clean_data function
from extract_data import connect_to_db  # Assuming you have the function to fetch data from DB

def perform_bivariate_analysis(df):
    # Define the output directory for saving plots
    output_dir_png = '../outputs/'
    os.makedirs(output_dir_png, exist_ok=True)

    # Ensure the 'total_dl_ul_data' column exists and is calculated
    df['total_dl_ul_data'] = df['total_download_data'] + df['total_upload_data']  # Sum of DL and UL data

    # List of application volume columns to analyze
    applications = [
        'social_media_volume', 'google_volume', 'email_volume', 
        'youtube_volume', 'netflix_volume', 'gaming_volume', 'other_volume'
    ]
    
    # Iterate through each application and analyze the relationship
    for app in applications:
        if app in df.columns:  # Check if the application column exists in the DataFrame
            print(f"\nBivariate Analysis - {app} vs Total DL+UL Data:")

            # Scatter plot for the relationship between the application volume and total DL+UL data
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=app, y='total_dl_ul_data', color='skyblue')
            plt.xlabel(f'{app} (Bytes)')
            plt.ylabel('Total Download + Upload Data (Bytes)')
            plt.title(f'Relationship between {app} and Total DL+UL Data')
            plt.grid(True)
            plt.tight_layout()

            # Save and display the plot
            plt.savefig(os.path.join(output_dir_png, f'bivariate_{app}_data.png'))
            plt.show()

            # Calculate and print the correlation between the application volume and total DL+UL data
            correlation = df[app].corr(df['total_dl_ul_data'])
            print(f"Correlation between {app} and Total DL+UL Data: {correlation:.4f}")

    return df

if __name__ == "__main__":
    # Fetch data from the database using the function from extract_data.py
    df = connect_to_db()

    if df is not None:
        # Clean the data using the function from clean_data.py
        df = clean_data(df)
        
        # Perform bivariate analysis
        df = perform_bivariate_analysis(df)

        # Optionally print summary of the results if needed
        print("\nBivariate analysis completed successfully.")
    else:
        print("Failed to fetch data.")
