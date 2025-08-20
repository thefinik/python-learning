import pandas as pd

# Create the DataFrame
data = pd.DataFrame({
    'name': ['Olya', 'Max', 'Solomiya', 'Andriy', 'Lina', 'Bohdan', 'Nastya'],
    'grade': [70, 85, 90, 60, 88, 73, 95]
})

# Step 1: Filter students with grade > 75
good_students = data[data['grade'] > 75]

# Step 2: Sort them in descending order of grade
sorted_students = good_students.sort_values(by='grade', ascending=False)

# Step 3: Show the top 3 students
print(sorted_students.head(3))