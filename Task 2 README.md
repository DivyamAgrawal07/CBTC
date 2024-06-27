
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
data_1 = pandas.read_csv('Unemployment in India - Unemployment in India.csv')
data_2 = pandas.read_csv('Unemployment_Rate_upto_11_2020 - Unemployment_Rate_upto_11_2020.csv')

# Display the first few rows of both datasets
print("Dataset 1 Preview:")
print(data_1.head())

print("\nDataset 2 Preview:")
print(data_2.head())

# Display basic information about the datasets
print("\nDataset 1 Info:")
print(data_1.info())

print("\nDataset 2 Info:")
print(data_2.info())

# Check for missing values in both datasets
print("\nMissing Values in Dataset 1:")
print(data_1.isnull().sum())

print("\nMissing Values in Dataset 2:")
print(data_2.isnull().sum())

# Cleaning and preprocessing (if necessary)
# Here we can perform tasks like handling missing values, converting data types, etc.

# For example, if there are missing values, we can fill or drop them
data_1 = data_1.dropna()
data_2 = data_2.dropna()

# Convert date columns to datetime if necessary
data_1['Date'] = pandas.to_datetime(data_1['Date'])
data_2['Date'] = pandas.to_datetime(data_2['Date'])

# Analyzing unemployment rate trends
# We can group the data by date and calculate the average unemployment rate

# Assuming 'Date' and 'Unemployment_Rate' are the relevant columns
data_1_grouped = data_1.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
data_2_grouped = data_2.groupby('Date')['Estimated Unemployment Rate (%)'].mean()

# Visualization
# Plotting the unemployment rate trends

plt.figure(figsize=(14, 7))
sns.lineplot(data=data_1, x='Date', y='Estimated Unemployment Rate (%)', label='Dataset 1')
sns.lineplot(data=data_2, x='Date', y='Estimated Unemployment Rate (%)', label='Dataset 2')
plt.title('Unemployment Rate Trends')
plt.xlabel('Date')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Additional analysis and visualization can be performed as needed
