import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from extract_data import connect_to_db  # Import the function to fetch data

# Get the current directory where the script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the absolute path for the outputs folder
output_folder = os.path.join(current_directory, "../outputs")

# Create the outputs folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load data using the function from extract_data.py
df = connect_to_db()

if df is not None:
    # Check if 'handset_type' exists in the DataFrame
    if 'handset_type' not in df.columns or 'handset_manufacturer' not in df.columns:
      print("Error: Required columns ('handset_type' or 'handset_manufacturer') not found in DataFrame.")
      print("Available columns:", df.columns)
    else:
        # 1. Identify Top 10 Handsets Used
        top_10_handsets = df['handset_type'].dropna().value_counts().head(10)  # dropna to handle None/NaN values
        print("Top 10 Handsets Used:")
        print(top_10_handsets)

        # Plotting the top 10 handsets for visualization
        plt.figure(figsize=(10, 6))
        top_10_handsets.plot(kind='bar', color='skyblue')
        plt.title('Top 10 Handsets Used by Customers')
        plt.xlabel('Handset')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        handset_plot_path = os.path.join(output_folder, f"top_10_handsets_{timestamp}.png")
        print(f"Saving plot: {handset_plot_path}")
        plt.savefig(handset_plot_path)
        plt.close()  # Close the plot to free up memory

        # 2. Identify Top 3 Manufacturers
        # Extract manufacturer names from the 'handset_manufacturer' column
        df['manufacturer'] = df['handset_manufacturer'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Unknown')  # Handle None/NaN values

        # Top 3 manufacturers
        top_3_manufacturers = df['manufacturer'].value_counts().head(3)
        print("Top 3 Manufacturers:")
        print(top_3_manufacturers)

        # Plotting the top 3 manufacturers for visualization
        plt.figure(figsize=(8, 6))
        top_3_manufacturers.plot(kind='bar', color='lightgreen')
        plt.title('Top 3 Handset Manufacturers')
        plt.xlabel('Manufacturer')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Save the plot with a unique filename
        manufacturer_plot_path = os.path.join(output_folder, f"top_3_manufacturers_{timestamp}.png")
        print(f"Saving plot: {manufacturer_plot_path}")
        plt.savefig(manufacturer_plot_path)
        plt.close()

        # 3. Top 5 Handsets per Top 3 Manufacturers
        print("\nTop 5 Handsets per Top 3 Manufacturers:")
        for manufacturer in top_3_manufacturers.index:
            top_5_handsets = df[df['manufacturer'] == manufacturer]['handset_type'].dropna().value_counts().head(5)  # dropna to handle None/NaN values
            print(f"\nTop 5 handsets for {manufacturer}:")
            print(top_5_handsets)

            # Optional: You can plot the top 5 handsets per manufacturer
            plt.figure(figsize=(10, 6))
            top_5_handsets.plot(kind='bar', color='lightcoral')
            plt.title(f'Top 5 Handsets for {manufacturer}')
            plt.xlabel('Handset')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save the plot with a unique filename
            handset_per_manufacturer_plot_path = os.path.join(output_folder, f"top_5_handsets_{manufacturer}_{timestamp}.png")
            print(f"Saving plot: {handset_per_manufacturer_plot_path}")
            plt.savefig(handset_per_manufacturer_plot_path)
            plt.close()

        # 4. Interpretation and Recommendations for Marketing Teams

        # Example recommendations:
        recommendations = """
        1. Focus Marketing on Popular Devices: 
           Marketing campaigns can be optimized to target users of the most popular handset models and manufacturers, as they represent the majority of the user base.

        2. Device-Specific Promotions: 
           Design promotional offers or partnerships with top manufacturers to attract more customers, especially those using the top 5 handsets per manufacturer.

        3. Network Optimization for Popular Devices: 
           Ensure the network infrastructure is optimized for the most commonly used handsets to enhance user experience, reducing complaints related to device compatibility.

        4. Upselling Opportunities: 
           Promote high-data-usage packages to users of high-end devices, as they are more likely to engage with data-intensive applications such as video streaming and gaming.
        """

        print("\nInterpretation and Recommendations for Marketing Teams:")
        print(recommendations)
else:
    print("Failed to fetch data.")
