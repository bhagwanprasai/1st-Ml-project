import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Read the CSV file into a DataFrame
data = pd.read_csv('E:/python/1st project/dataset/housing.csv')

# Convert 'yes'/'no' columns to binary (1/0)
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# Separate features (X) and target variable (y)
X = data.drop('price', axis=1)
y = data['price']

# One-Hot Encode the 'furnishingstatus' column
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
furnishing_encoded = ohe.fit_transform(X[['furnishingstatus']])
furnishing_encoded_df = pd.DataFrame(furnishing_encoded.toarray(), columns=ohe.get_feature_names_out(['furnishingstatus']))

# Concatenate encoded columns with the rest of the features
X = pd.concat([X.drop('furnishingstatus', axis=1), furnishing_encoded_df], axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = model.predict(X_test)

# --- Analysis and Visualization ---

# 1. Distribution of Actual Prices
actual_prices_chart = alt.Chart(pd.DataFrame({'Actual Prices': y})).mark_bar().encode(
    x=alt.X('Actual Prices:Q', bin=True),
    y=alt.Y('count()', title='Frequency'),
    tooltip=['Actual Prices:Q', 'count()']
).properties(
    title='Distribution of Actual House Prices'
).interactive()

actual_prices_chart.save('actual_prices_distribution.json')

# 2. Distribution of Predicted Prices
predicted_prices_chart = alt.Chart(pd.DataFrame({'Predicted Prices': y_pred})).mark_bar().encode(
    x=alt.X('Predicted Prices:Q', bin=True),
    y=alt.Y('count()', title='Frequency'),
    tooltip=['Predicted Prices:Q', 'count()']
).properties(
    title='Distribution of Predicted House Prices'
).interactive()

predicted_prices_chart.save('predicted_prices_distribution.json')

# 3. Scatter Plot of Actual vs. Predicted Prices (Altair)
scatter_chart = alt.Chart(pd.DataFrame({'Actual Prices': y_test, 'Predicted Prices': y_pred})).mark_circle().encode(
    x='Actual Prices',
    y='Predicted Prices',
    tooltip=['Actual Prices', 'Predicted Prices']
).properties(
    title='Actual vs. Predicted House Prices (Altair)'
).interactive()

# Add a diagonal reference line
scatter_chart += scatter_chart.mark_line(color='red', strokeDash=[5, 5]).encode(
    x='Actual Prices',
    y='Actual Prices'
)

scatter_chart.save('actual_vs_predicted_scatter_altair.json')

#display of chart in html format
scatter_chart.save('actual_price_distribution.html')
scatter_chart.save('actual_vs_predictied_scatter.html')
scatter_chart.save('predicted_prices_distribution.html')
# Print the R-squared value of the model
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")
