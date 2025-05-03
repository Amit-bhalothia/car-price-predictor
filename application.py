import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


app = Flask(__name__)

# Load the saved Random Forest model (which is a Pipeline in this case)
model = pickle.load(open('RandomForestModel.pkl', 'rb'))

# Load the dataset
df = pd.read_csv('dataset.csv')

# Prepare the logos dynamically by reading from a folder
companies = df['Company'].unique().tolist()

# Create a dictionary for models based on the company
car_models_dict = {}
for company in companies:
    car_models_dict[company] = df[df['Company'] == company]['Model'].unique().tolist()


# Generate and save the decision tree visualization to show how the random forest works
def save_decision_tree_image():
    image_path = 'static/decision_tree.png'

    if not os.path.exists(image_path):
        # Extract the Random Forest model from the pipeline
        random_forest_model = model.named_steps['randomforestregressor']

        # Select the first tree for simplicity otherwise it is hard to read the image
        tree = random_forest_model.estimators_[0]

        # Feature names after column transformation
        feature_names = ['Model', 'Company', 'Year', 'kms_driven', 'fuel_type', 'transmission']

        # Extract feature names used in the decision tree
        feature_names_transformed = feature_names
        if hasattr(model.named_steps['columntransformer'], 'transformers_'):
            # Extract feature names from the ColumnTransformer
            the_feature_names = model.named_steps['columntransformer'].transformers_[0][1].get_feature_names_out()
            feature_names_transformed = the_feature_names.tolist() + ['Year', 'kms_driven']

        # plot the figure
        plt.figure(figsize=(20, 10))
        plot_tree(tree, filled=True, feature_names=feature_names_transformed, max_depth=5, fontsize=8)
        plt.title('Decision Tree from Random Forest')
        plt.savefig(image_path)  # Save the plot as a .png file
        plt.close()


save_decision_tree_image()


# Route for the main page as home page
@app.route('/')
def index():
    return render_template('index.html', companies=companies, models=car_models_dict)


# fetch the About Us page of the website
@app.route('/about')
def about():
    return render_template('about.html')


# fetch the Car Companies page of the website
@app.route('/car-companies')
def car_companies():
    return render_template('car-companies.html', companies=companies)


# Route to fetch models dynamically based on selected company
@app.route('/get_models', methods=['POST'])
def get_models():
    selected_company = request.form['company']
    models = car_models_dict.get(selected_company, [])
    return jsonify(models)


# Prediction route with metrics calculation (R², MAE, etc.)
@app.route('/predict', methods=['POST'])
def predict():
    Car_company = request.form['Company']
    Car_model = request.form['Model']
    year = int(request.form['Year'])
    Kms_driven = int(request.form['kms_driven'])
    fuel_type = request.form['fuel_type']
    transmission = request.form['transmission']

    # Prepare the input data for the model
    input_data = pd.DataFrame([[Car_model, Car_company, year, Kms_driven, fuel_type, transmission]],
                              columns=['Model', 'Company', 'Year', 'kms_driven', 'fuel_type', 'transmission'])

    # Make prediction using the trained Random Forest model
    predicted_price_in_INR = model.predict(input_data)[0]

    # Convert predicted price INR to GBP
    Exchange_rate = 100
    price_in_gbp = (predicted_price_in_INR * 100000) / Exchange_rate

    # Evaluation Metrics
    X = df[['Model', 'Company', 'Year', 'kms_driven', 'fuel_type', 'transmission']]
    y = df['Price']  # Assuming 'Price' is the target variable in your dataset
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # Return predicted price and metrics
    return jsonify({
        'price': f"{price_in_gbp:.2f} GBP",
        'r2': f"{r2:.2f}",
        'mae': f"{mae:.2f}",
        'mse': f"{mse:.2f}"
    })


if __name__ == '__main__':
    app.run(debug=True)
