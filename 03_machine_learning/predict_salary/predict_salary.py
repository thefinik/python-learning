import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
data = pd.read_csv("03_machine_learning/predict_salary/salary_data.csv")
X = data[['experience']]
y = data['salary']

# 2. Create and train the model
model = LinearRegression()
model.fit(X, y)

# 3. Make predictions
experience = np.array([[1.5], [3.5], [5.0]])
prediction = model.predict(experience)

for i in range(len(experience)):
    print(f"{experience[i][0]} years → ${prediction[i]:,.2f}")

# 4. Visualization
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='green', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction Based on Experience')
plt.legend()
plt.grid(True)
plt.savefig("chart.png")
plt.show()