import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Loading the dataset
df = pd.read_csv('uncleaned_data.csv')


# Dropped the unwanted column
df = df.drop(['Unnamed: 0'], axis=1)

# Rename all the columns
df = df.rename(
    columns={'model': 'Model', 'year': 'Year', 'price': 'Price', 'fuelType': 'fuel_type', 'mileage': 'kms_driven'})

df.head()
df.info()
df.describe()

# Dropping all the nan values
df = df.dropna()
# Dropping  all the duplicates
df = df.drop_duplicates()

# Remove all the outliner's in the dataset
df = df[df['Year'] >= 2008]
df = df[df['Year'] <= 2024]
df = df[df['kms_driven'] <= 140000]
df = df[df['kms_driven'] > 1000]
df = df[df['transmission'] != 'Other']
df['Price'] = (df['Price'] / 1000)
df = df[df['Price'] <= 130]
df = df[df['fuel_type'] != 'Electric']
df = df[df['fuel_type'] != 'Other']
df = df[~((df['Price'] > 80) & (df['Company'] == 'Hyundai'))]
df = df[~((df['Price'] > 80) & (df['Company'] == 'Skoda'))]
df = df[~((df['Price'] > 80) & (df['Company'] == 'BMW'))]
df = df[~((df['Price'] > 40) & (df['Company'] == 'Ford'))]
df = df[~((df['Price'] > 40) & (df['Company'] == 'Vauxhall'))]

model_counts = df['Model'].value_counts()

# Filter out models with counts less than 30
models_to_keep = model_counts[model_counts >= 100].index

# Drop rows where the model is not in the list of models to keep
df = df[df['Model'].isin(models_to_keep)]

df['Price'] = (df['Price'] / 1000)


# Converting a column to integer
df['kms_driven'].astype('int')


# Create a scatter plot with regression line
plt.figure(figsize=(12, 7))

# Price vs Kms_driven with Regression Line
sns.regplot(data=df, x='kms_driven', y='Price', scatter_kws={'s':100, 'alpha':0.7}, line_kws={'color':'red', 'lw':2})

plt.title('Effect of kms_driven on Price')
plt.xlabel('kms_driven')
plt.ylabel('Price')
plt.grid(True)

plt.show()

# Price vs Year with Regression Line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Price', y='Year', scatter_kws={'s':100}, line_kws={'color':'red'})
plt.title('Price vs Year with Regression Line')
plt.xlabel('Price')
plt.ylabel('Year')
plt.show()

# strip plot Price vs Company
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='Company', y='Price', palette='Set2', jitter=True)
plt.title('Price Distribution by Company (Strip Plot)')
plt.xlabel('Company')
plt.ylabel('Price')
plt.show()


