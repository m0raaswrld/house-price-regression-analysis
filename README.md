Predicting house prices using regression analysis
### Overview
This project aims to predict house prices using regression analysis, leveraging key features such as size, location, number of rooms, and additional engineered features. By exploring and modeling real-world data, this project showcases the application of core data science concepts, including data preprocessing, feature engineering, model training, and evaluation.

We utilized Python and its robust ecosystem of libraries, such as pandas, scikit-learn, matplotlib, and seaborn, for analysis, visualization, and building predictive models.

The project emphasizes the following:
	•	Comprehensive data exploration and cleaning.
	•	Advanced feature engineering to create predictive variables.
	•	Evaluation and comparison of machine learning models.
	•	Deployment of the final model using Flask for a web-based prediction interface.

### Key Features
- *Data Cleaning*: Handled missing values and removed outliers for better quality.
- *Feature Engineering*: Derived new metrics such as price per square foot.
- *Model Building*: Implemented and evaluated regression models with scikit-learn.
- *Visualization*: Highlighted key trends and results using interactive charts.

---

### Group Members
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
### Checking for missing values
missing_values = data.isnull().sum()
print(missing_values)

### Imputing missing values with mean
for feature in ['sqft', 'bedrooms', 'bathrooms']:
    data[feature].fillna(data[feature].mean(), inplace=True)

Removing Outliers

To ensure the dataset’s quality, we addressed extreme outliers using interquartile range (IQR):

### Removing outliers based on price
Q1 = data['price'].quantile(0.25)
Q3 = data['price'].quantile(0.75)
IQR = Q3 - Q1

Key Visualizations
### House Price Distribution
A histogram revealed the skewness in house prices, guiding our transformation strategy.

### import matplotlib module
from matplotlib import pyplot as plt
plt.hist(data['price'], bins=50, color='blue', alpha=0.7) 
plt.title('House Price Distribution') 
plt.xlabel('Price') 
plt.ylabel('Frequency') 
plt.show()
### Feature Correlation Heatmap
The heatmap uncovered relationships between numerical features.

pip install seaborn
import seaborn as sns
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

### Feature engineering is crucial for enhancing predictive power.
We created additional features such as price_per_sqft to capture the impact of property size on pricing.

data['price_per_sqft'] = data['price'] / data['sqft']

### Feature Selection
To identify the most relevant features, we performed correlation analysis and used Variance Inflation Factor (VIF) to eliminate multicollinearity:
from statsmodels.stats.outliers_influence import variance_inflation_factor

### Calculating VIF for each feature
X = data[['sqft', 'bedrooms', 'bathrooms', 'location_score']]
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

 Model Building

### Model Selection
We trained multiple models, including Linear Regression, Random Forest, and Gradient Boosting, and selected the best-performing one based on evaluation metrics.
Training the Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

### Splitting data into train and test sets
X = data[['sqft', 'bedrooms', 'bathrooms', 'price_per_sqft']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Training the model
model = LinearRegression()
model.fit(X_train, y_train)

### Model Evaluation

### Predictions and metrics
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-Squared:", r2_score(y_test, y_pred))

### Key Visualizations
•Prediction vs. Actual Scatter Plot
This scatter plot highlights how well the model predicts house prices.

plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.title('Predicted vs. Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

### Deployment
Deployment Using Flask
The final model was deployed using Flask to provide an accessible web interface for predictions.
Flask API Implementation
from flask import Flask, request, jsonify

import joblib
app = Flask(_name_)

### Load trained model
model = joblib.load('linear_regression_model.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})
if _name_ == '_main_':
    app.run(debug=True)

### Limitations and Future Enhancements
Limitations
-The model performance could be improved by incorporating additional features like proximity to amenities or crime rates.
-Linear regression may not capture complex relationships in the data.
-Future Enhancements
-Experimenting with advanced models like Random Forest, XGBoost, or Neural Networks.
-Conducting hyperparameter tuning to optimize model performance.
-Building an interactive dashboard for data visualization and prediction insights.

### Conclusion
This project successfully predicted house prices using regression analysis, demonstrating a structured approach to data science. By integrating exploratory data analysis, feature engineering, model building, and deployment, we created a reproducible and scalable pipeline for house price prediction.


