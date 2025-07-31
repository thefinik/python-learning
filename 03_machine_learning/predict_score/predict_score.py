import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load the data
data = pd.read_csv("03_machine_learning/predict_score/study_data.csv")
X = data[['hours']]
y = data['score']

# 2. Create and train the model
model = LinearRegression()
model.fit(X, y)

# 3. Make predictions
hours = np.array([[4.5], [6.5], [9.5]])
prediction = model.predict(hours)

for i in range(len(hours)):
    print(f"{hours[i][0]} hours → {prediction[i]:.2f} points")

# 4. Visualization
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours of Study')
plt.ylabel('Score')
plt.title('Score Prediction Based on Study Time')
plt.legend()
plt.grid(True)
plt.savefig("chart.png")
plt.show()