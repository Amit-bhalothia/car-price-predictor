import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

# Load the data as csv file
df = pd.read_csv('dataset.csv')

# Drop unwanted columns that are not useful in prediction
df = df.drop(['Unnamed: 0'], axis=1)

# Independent features as input data
X = df.drop(columns='Price')

# Dependent variable as target column
y = df['Price']

# Split the data into training (80%) and testing (20%) sets so that I can check the effectiveness of th emodel
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize OneHotEncoder to convert categorical column into numerical column
ohe = OneHotEncoder(handle_unknown='ignore')

# Fit the OneHotEncoder on the relevant columns in the training data
ohe.fit(X_train[['Model', 'Company', 'fuel_type', 'transmission']])

# Create the column transformer for the selective columns only
column_trans = make_column_transformer(
    (ohe, ['Model', 'Company', 'fuel_type', 'transmission']),
    remainder='passthrough'  # This will pass through other columns without transformation
)

# The Random Forest Regressor model
rf = RandomForestRegressor()

# Create a pipeline
pipe = make_pipeline(column_trans, rf)

# Fit the pipeline on the training data so that later on we can compare the result with testing data
pipe.fit(X_train, y_train)

# Predict on the test data passing through the pipeline
y_pred = pipe.predict(X_test)

# Print the R-squared score as accuracy
print("The R² Score is: ", r2_score(y_test, y_pred))

# Save the model to a file so that I can use in application
pickle.dump(pipe, open('RandomForestModel.pkl', 'wb'))

