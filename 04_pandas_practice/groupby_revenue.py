import pandas as pd

# Create the DataFrame
data = pd.DataFrame({
    'city': ['Kyiv', 'Lviv', 'Kyiv', 'Odesa', 'Lviv', 'Odesa', 'Kyiv'],
    'product': ['apple', 'banana', 'banana', 'orange', 'apple', 'orange', 'apple'],
    'quantity': [10, 5, 7, 8, 6, 5, 4],
    'price': [2, 1, 1, 3, 2, 3, 2]
})

# Step 1: Calculate revenue (price * quantity)
data['revenue'] = data['quantity'] * data['price']

# Step 2: Group by city and sum revenue
revenue_by_city = data.groupby('city')['revenue'].sum()
print("Revenue by city:")
print(revenue_by_city)

# Step 3: Group by product and sum revenue
revenue_by_product = data.groupby('product')['revenue'].sum()
print("\nRevenue by product:")
print(revenue_by_product)