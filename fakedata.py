import pandas as pd
import numpy as np

# Load the Excel files
adult_file_path = 'adult.xlsx'
salary_datafields_file_path = 'salary_datafields.xlsx'

# Load the datasets
adult_data = pd.read_excel(adult_file_path)
salary_datafields = pd.read_excel(salary_datafields_file_path)

# Identify categorical and numerical columns based on salary_datafields
categorical_columns = salary_datafields[salary_datafields['Type'] == 'Categorical']['Variable Name'].tolist()
numerical_columns = salary_datafields[salary_datafields['Type'] == 'Integer']['Variable Name'].tolist()

# Ensure 'sex' column is included in the categorical columns list explicitly
if 'sex' not in categorical_columns:
    categorical_columns.append('sex')

# Remove 'fnlwgt' and 'income' from the column lists if present
if 'fnlwgt' in numerical_columns:
    numerical_columns.remove('fnlwgt')
if 'income' in categorical_columns:
    categorical_columns.remove('income')

# For categorical columns, get the unique values and their frequencies
categorical_values_dist = {col: adult_data[col].value_counts(normalize=True).to_dict() for col in categorical_columns}

# For numerical columns, get the min, max, and mean
numerical_stats = adult_data[numerical_columns].describe().T[['min', 'max', 'mean']]

# Updated function to generate fake data with more accurate ranges and distribution
def generate_improved_fake_data(n_samples):
    fake_data = {}
    # Generate fake data for categorical columns based on real data distribution
    for col, value_dist in categorical_values_dist.items():
        fake_data[col] = np.random.choice(list(value_dist.keys()), size=n_samples, p=list(value_dist.values()))
    
    # Generate fake data for numerical columns, keeping the values within a realistic range
    for col, stats in numerical_stats.iterrows():
        min_val, max_val, mean_val = stats['min'], stats['max'], stats['mean']
        
        # Generate normal distributed data but clamp it within min/max values
        fake_data[col] = np.clip(np.random.normal(mean_val, (max_val - min_val) / 6, size=n_samples), min_val, max_val).astype(int)
    
    # Convert to DataFrame
    fake_df = pd.DataFrame(fake_data)
    return fake_df

# Ask the user for the number of fake data entries
while True:
    try:
        n_samples = int(input("How many fake data entries do you want to generate? "))
        if n_samples > 0:
            break
        else:
            print("Please enter a positive integer.")
    except ValueError:
        print("Invalid input. Please enter a positive integer.")

# Generate fake data based on user input
fake_data = generate_improved_fake_data(n_samples)

# Save the generated fake data to a CSV file
fake_data.to_csv('fake_data.csv', index=False)

# Display the first 10 rows of the generated data
print(fake_data.head(10))

# Print the total number of generated entries
print(f"\nTotal number of fake data entries generated: {len(fake_data)}")