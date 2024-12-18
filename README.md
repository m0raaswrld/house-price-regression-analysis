Predicting house prices using regression analysis

## Overview
This project focuses on building a predictive model for house prices based on features like size, location, and room count. Utilizing Python and machine learning libraries, we explored regression techniques, emphasizing clean data and insightful visualizations.

### Key Features
- *Data Cleaning*: Handled missing values and removed outliers for better quality.
- *Feature Engineering*: Derived new metrics such as price per square foot.
- *Model Building*: Implemented and evaluated regression models with scikit-learn.
- *Visualization*: Highlighted key trends and results using interactive charts.

---

## Group Members
- Sylvia
- Mitchelle
- Lavendar

---

## Installation

### Prerequisites
Ensure the following are installed on your machine:
- Python 3.8 or higher
- Jupyter Notebook or any IDE
- Git

### Steps
1. *Clone the Repository*:
   ```bash
   git clone https://github.com/your-username/house-prices-regression.git
   cd house-prices-regression
pip install -r requirements.txt
### Data cleaning
missing_values = data.isnull().sum()
data['feature'].fillna(data['feature'].mean(), inplace=True)

### Feature Engineering
data['price_per_sqft'] = data['price'] / data['sqft']
import seaborn as sns
sns.heatmap(data.corr(), annot=True)

### Model Building
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

### Deployment
from flask import Flask, request, jsonify
import joblib

app = Flask(_name_)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if _name_ == '_main_':
    app.run(debug=True)
