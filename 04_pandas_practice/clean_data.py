import pandas as pd

# Create the DataFrame
data = pd.DataFrame({
    'product': ['milk', 'corn', 'butter', 'jam'],
    'price': [25, None, 30, None],
    'quantity': ['2', '7', None, '11']
})

# Step 1: Check for missing values
print("Missing values:")
print(data.isnull().sum())

# Step 2: Fill missing values with 0
data['price'] = data['price'].fillna(0)
data['quantity'] = data['quantity'].fillna('0')

# Step 3: Drop duplicates if any
data = data.drop_duplicates()

# Step 4: Display the cleaned data
print("\nCleaned data:")
print(data)