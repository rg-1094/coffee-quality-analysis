# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# Note: In practice, you would use the full dataset from Kaggle
# For this example, I'll use the sample data provided
data = pd.read_csv("YourDirectory")

# Display basic info about the dataset
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Data Cleaning and Preprocessing

# Select relevant columns for our analysis
# Based on the sample data, these are the most relevant features
selected_columns = [
    'Country.of.Origin', 'Variety', 'Processing.Method', 'Aroma', 'Flavor', 
    'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 
    'Sweetness', 'Cupper.Points', 'Total.Cup.Points', 'Moisture', 
    'Category.One.Defects', 'Quakers', 'altitude_mean_meters'
]

# Create a working dataframe with selected columns
coffee_df = data[selected_columns].copy()

# Check for missing values
print("\nMissing values per column:")
print(coffee_df.isnull().sum())

# Handle missing values
# For numerical columns, we'll fill with median
numerical_cols = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 
                 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points', 
                 'Total.Cup.Points', 'Moisture', 'Category.One.Defects', 
                 'Quakers', 'altitude_mean_meters']

imputer = SimpleImputer(strategy='median')
coffee_df[numerical_cols] = imputer.fit_transform(coffee_df[numerical_cols])

# For categorical columns, fill with mode
categorical_cols = ['Country.of.Origin', 'Variety', 'Processing.Method']
for col in categorical_cols:
    mode_value = coffee_df[col].mode()[0]
    coffee_df[col] = coffee_df[col].fillna(mode_value)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    coffee_df[col] = le.fit_transform(coffee_df[col].astype(str))
    label_encoders[col] = le  # Save encoders for reference

# Verify preprocessing
print("\nAfter preprocessing:")
print(coffee_df.head())
print("\nMissing values after preprocessing:")
print(coffee_df.isnull().sum())

# Clustering Analysis - Grouping coffees by flavor characteristics

# Select features for clustering - using sensory attributes
flavor_features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Sweetness']
X_cluster = coffee_df[flavor_features]

# Standardize the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Determine optimal number of clusters using Elbow Method
wcss = []
silhouette_scores = []
cluster_range = range(2, 6)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)

# Plot Elbow Method and Silhouette Scores
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')

plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Based on the plots, select optimal number of clusters
optimal_clusters = 3  # Adjust based on your analysis of the plots

# Perform K-Means clustering with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
coffee_df['Cluster'] = clusters

# Analyze cluster characteristics
cluster_summary = coffee_df.groupby('Cluster')[flavor_features].mean()
print("\nCluster Characteristics (Mean Values):")
print(cluster_summary)

# Visualize clusters using PCA for dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=coffee_df['Cluster'], 
                palette='viridis', s=100)
plt.title('Coffee Clusters by Flavor Profiles')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Regression Analysis - Predicting Total Cup Points

# Select features for regression
# Using both numerical and categorical features that might affect quality
regression_features = ['altitude_mean_meters', 'Processing.Method', 'Country.of.Origin',
                      'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
X_reg = coffee_df[regression_features]
y_reg = coffee_df['Total.Cup.Points']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions and evaluate model
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\nRegression Model R-squared Score: {r2:.2f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': regression_features,
    'Coefficient': regressor.coef_
}).sort_values('Coefficient', ascending=False)

print("\nFeature Importance (Coefficients):")
print(feature_importance)

# Visualize actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
plt.xlabel('Actual Total Cup Points')
plt.ylabel('Predicted Total Cup Points')
plt.title('Actual vs Predicted Coffee Ratings')
plt.show()

# Residual analysis
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (Actual - Predicted)')
plt.show()

# Save the processed data and results for reference
coffee_df.to_csv('YourDirectory', index=False)

# Save the models (in practice, you would use pickle/joblib)
# Example:
# import joblib
# joblib.dump(kmeans, 'coffee_cluster_model.pkl')
# joblib.dump(regressor, 'coffee_rating_predictor.pkl')