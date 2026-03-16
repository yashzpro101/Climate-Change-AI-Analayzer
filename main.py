import matplotlib
matplotlib.use('Qt5Agg')  # Changed from TkAgg to Qt5Agg
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Load the scientific data
try:
    df = pd.read_csv('climate_data.csv')
    print("Successfully loaded NASA climate data.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 2. Prepare the AI Model (Linear Regression)
# We use 'Year' to predict 'Temperature Anomaly'
X = df[['Year']].values
y = df['J-D'].values

model = LinearRegression()
model.fit(X, y) # This 'trains' the AI

# 3. Predict the future (e.g., Year 2030)
future_year = np.array([[2030]])
prediction = model.predict(future_year)

# 4. Create a Professional Visualization
plt.style.use('seaborn-v0_8-darkgrid') # Makes the graph look modern
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Recorded Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='ML Trend Line')

plt.title('Climate Change Analysis: AI Prediction Model', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temp Anomaly (°C)', fontsize=12)
plt.legend()

# Show the results
print(f"Based on current trends, the 2030 anomaly is predicted to be: {prediction[0]:.2f}°C")
plt.show()