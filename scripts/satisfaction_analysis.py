import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import pymysql
from extract_data import connect_to_db
from sklearn.linear_model import LinearRegression

# Load data from the database
df = connect_to_db()

# Columns that may be useful for engagement and experience analysis
engagement_columns = ['total_download_data', 'total_upload_data', 'total_session_duration']
experience_columns = ['tcp_dl_retransmission', 'tcp_ul_retransmission', 'avg_rtt_dl', 'avg_rtt_ul']

# Check for and handle NaN values in the engagement and experience columns
df[engagement_columns] = df[engagement_columns].fillna(df[engagement_columns].mean())
df[experience_columns] = df[experience_columns].fillna(df[experience_columns].mean())

# Calculate the engagement score (Euclidean distance from "less engaged" cluster)
less_engaged_center = np.array([df[engagement_columns].mean()])  # Using mean of engagement columns
df['engagement_score'] = euclidean_distances(df[engagement_columns], less_engaged_center).flatten()

# Calculate the experience score (Euclidean distance from "worst experience" cluster)
worst_experience_center = np.array([df[experience_columns].mean()])  # Using mean of experience columns
df['experience_score'] = euclidean_distances(df[experience_columns], worst_experience_center).flatten()

# Calculate the satisfaction score as the average of engagement and experience scores
df['satisfaction_score'] = (df['engagement_score'] + df['experience_score']) / 2

# Task 4.2 - Top 10 Satisfied Customers
top_10_satisfied = df[['user_id', 'satisfaction_score']].sort_values(by='satisfaction_score', ascending=False).head(10)
print("Top 10 Satisfied Customers:")
print(top_10_satisfied)

# Task 4.3 - Build a regression model to predict satisfaction score
X = df[['engagement_score', 'experience_score']]
y = df['satisfaction_score']
regressor = LinearRegression()
regressor.fit(X, y)

# Print the coefficients and intercept of the regression model
print(f"Regression Coefficients: {regressor.coef_}")
print(f"Intercept: {regressor.intercept_}")

# Task 4.4 - KMeans Clustering on engagement and experience scores (k=2)
kmeans = KMeans(n_clusters=2, random_state=42)
df['satisfaction_cluster'] = kmeans.fit_predict(df[['engagement_score', 'experience_score']])

# Task 4.5 - Aggregate satisfaction and experience scores per cluster
cluster_scores = df.groupby('satisfaction_cluster')[['satisfaction_score', 'experience_score']].mean()
print("Average Satisfaction and Experience Scores per Cluster:")
print(cluster_scores)

# Task 4.6 - Export to MySQL
try:
    conn = pymysql.connect(host='localhost', user='root', password='password', db='customer_satisfaction', charset='utf8mb4')
    cursor = conn.cursor()

    # Create table if not exists
    create_table_query = """
    CREATE TABLE IF NOT EXISTS user_satisfaction (
        user_id VARCHAR(255),
        engagement_score FLOAT,
        experience_score FLOAT,
        satisfaction_score FLOAT,
        satisfaction_cluster INT
    );
    """
    cursor.execute(create_table_query)

    # Insert data into table
    for _, row in df.iterrows():
        insert_query = """
        INSERT INTO user_satisfaction (user_id, engagement_score, experience_score, satisfaction_score, satisfaction_cluster)
        VALUES (%s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, (row['user_id'], row['engagement_score'], row['experience_score'], row['satisfaction_score'], row['satisfaction_cluster']))

    conn.commit()
    print("Data exported to MySQL successfully.")

except Exception as e:
    print(f"Error exporting data to MySQL: {e}")

finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()
