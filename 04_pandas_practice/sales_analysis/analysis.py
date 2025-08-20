import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("04_pandas_practice/sales_analysis/sales_data.csv")

# Convert price and units_sold to integers
data['price'] = data['price'].astype(int)
data['units_sold'] = data['units_sold'].astype(int)

# Calculate revenue
data['revenue'] = data['price'] * data['units_sold']

# Group by category and sum revenue
category_revenue = data.groupby('category')['revenue'].sum()

# Print result
print("Revenue by category:")
print(category_revenue)

# Optional: create a bar chart
category_revenue.plot(kind='bar', title='Revenue by Category')
plt.xlabel('Category')
plt.ylabel('Revenue')
plt.tight_layout()
plt.savefig('chart.png')
plt.show()