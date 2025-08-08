import pandas as pd

# Create the DataFrame
data = pd.DataFrame({
    'product': ['sunglasses', 'cases', 'bottles'],
    'price': ['30', '10', '15'],
    'quantity': ['3', '5', '4']
})

# Step 1: Convert string columns to integers
data['price'] = data['price'].astype(int)
data['quantity'] = data['quantity'].astype(int)

# Step 2: Create a new column for total revenue
data['total'] = data['price'] * data['quantity']

# Step 3: Show the result
print(data)