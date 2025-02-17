import numpy as np


# Ordinal Encoding
def ordinal_encode(categories, data):
    category_map = {category: index for index, category in enumerate(categories)}

    # Map each value in the data to its corresponding ordinal value
    encoded_data = [category_map[item] for item in data]

    return encoded_data



def one_hot_encode(categories, data):
    category_map = {category: index for index, category in enumerate(categories)}


    one_hot_encoded_data = []

    for item in data:
        # Create a zero vector for each category
        one_hot_vector = [0] * len(categories)

        # Set the position corresponding to the category to 1
        one_hot_vector[category_map[item]] = 1

        # Add the one-hot vector to the result list
        one_hot_encoded_data.append(one_hot_vector)

    return one_hot_encoded_data


# Example data for colors
color_categories = ['red', 'green', 'blue']  # Categories of colors
color_data = ['green', 'red', 'blue', 'blue', 'green']  # The data to be encoded

# Ordinal Encoding for colors
ordinal_encoded_colors = ordinal_encode(color_categories, color_data)
print("Ordinal Encoding (Colors):", ordinal_encoded_colors)

# One-Hot Encoding for colors
one_hot_encoded_colors = one_hot_encode(color_categories, color_data)
print("\nOne-Hot Encoding (Colors):")
for vector in one_hot_encoded_colors:
    print(vector)

# Example data for salary
salary_categories = ['low', 'medium', 'high']  # Categories of salary levels
salary_data = ['medium', 'low', 'high', 'medium', 'high']  # The data to be encoded

# Ordinal Encoding for salary
ordinal_encoded_salary = ordinal_encode(salary_categories, salary_data)
print("\nOrdinal Encoding (Salary):", ordinal_encoded_salary)

# One-Hot Encoding for salary
one_hot_encoded_salary = one_hot_encode(salary_categories, salary_data)
print("\nOne-Hot Encoding (Salary):")
for vector in one_hot_encoded_salary:
    print(vector)
