import matplotlib

matplotlib.use('Qt5Agg')  # Ensures a popup window on Windows 8.1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

"""
PROJECT: Statistical Analysis of Global Temperature Anomalies
AUTHOR: [Your Name]
PURPOSE: Modeling NASA GISTEMP data to predict climate trends via Linear Regression.
"""


def load_and_clean_data(filepath):
    """Loads CSV and removes non-numeric artifacts for statistical integrity."""
    try:
        df = pd.read_csv(filepath)

        # Coerce non-numeric data to NaN and drop missing values
        # This handles symbols like '***' or '?' found in raw climate datasets
        df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
        df = df.dropna(subset=['J-D'])

        print(f"[INFO] Data validation successful. Processing {len(df)} samples.")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None


def train_predictive_model(df):
    """Implements Linear Regression to identify temperature trends."""
    # Reshaping data for Scikit-Learn requirements
    X = df[['Year']].values
    y = df['J-D'].values

    # Initialize and train the Supervised Learning model
    model = LinearRegression()
    model.fit(X, y)

    return model, X, y


def visualize_results(X, y, model, prediction_2030):
    """Generates a high-resolution scientific plot of the findings."""
    plt.style.use('ggplot')  # Professional academic styling
    plt.figure(figsize=(12, 6))

    # Plotting historical observations
    plt.scatter(X, y, color='#3498db', label='NASA Recorded Anomalies', alpha=0.7)

    # Plotting the Machine Learning trend line
    plt.plot(X, model.predict(X), color='#e74c3c', linewidth=2, label='ML Linear Trend')

    # Formatting the chart for publication quality
    plt.title('Global Temperature Anomaly Analysis (1950 - Present)', fontsize=14)
    plt.xlabel('Calendar Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
    plt.legend(facecolor='white', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    print(f"[RESULT] Projected 2030 Temperature Anomaly: {prediction_2030[0]:.4f}°C")
    plt.show()


def main():
    # File configuration
    DATA_FILE = 'climate_data.csv'

    # Execute Pipeline
    data = load_and_clean_data(DATA_FILE)

    if data is not None:
        # Train model
        model, X, y = train_predictive_model(data)

        # Predict future value
        year_2030 = np.array([[2030]])
        prediction_2030 = model.predict(year_2030)

        # Output Visualization
        visualize_results(X, y, model, prediction_2030)


if __name__ == "__main__":
    main()
