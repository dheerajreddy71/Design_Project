import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import tkinter as tk
from tkinter import ttk

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
data = pd.read_csv('C:/Users/Windows/Downloads/pred_data.csv', encoding='ISO-8859-1')

# Data Preprocessing
data['arrival_date'] = pd.to_datetime(data['arrival_date'])
data['day'] = data['arrival_date'].dt.day
data['month'] = data['arrival_date'].dt.month
data['year'] = data['arrival_date'].dt.year
data.drop(['arrival_date'], axis=1, inplace=True)

# Define features and target variable
X = data.drop(['min_price', 'max_price', 'modal_price'], axis=1)
y = data[['min_price', 'max_price', 'modal_price']]

# Use OneHotEncoder to encode categorical variables
encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['state', 'district', 'market', 'commodity', 'variety'])
    ],
    remainder='passthrough'
)

X_encoded = encoder.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# GUI with Tkinter
def predict_prices():
    input_data = {
        'state': state_var.get(),
        'district': district_var.get(),
        'market': market_var.get(),
        'commodity': commodity_var.get(),
        'variety': variety_var.get(),
        'arrival_date': pd.to_datetime(arrival_date_var.get())
    }

    input_df = pd.DataFrame([input_data])
    input_df['day'] = input_df['arrival_date'].dt.day
    input_df['month'] = input_df['arrival_date'].dt.month
    input_df['year'] = input_df['arrival_date'].dt.year
    input_df.drop(['arrival_date'], axis=1, inplace=True)

    input_encoded = encoder.transform(input_df)

    # Make predictions
    predictions = model.predict(input_encoded)[0]

    # Update the output labels
    min_price_label.config(text=f"Min Price: {predictions[0]}")
    max_price_label.config(text=f"Max Price: {predictions[1]}")
    modal_price_label.config(text=f"Modal Price: {predictions[2]}")

# Create the Tkinter window
window = tk.Tk()
window.title("Price Prediction App")

# Create and place labels and entry widgets
state_label = ttk.Label(window, text="State:")
state_label.grid(row=0, column=0, padx=10, pady=10)
state_var = tk.StringVar()
state_entry = ttk.Entry(window, textvariable=state_var)
state_entry.grid(row=0, column=1, padx=10, pady=10)

# Similar setup for other input fields (district, market, commodity, variety, arrival_date)
# Create and place labels and entry widgets for other input fields
district_label = ttk.Label(window, text="District:")
district_label.grid(row=1, column=0, padx=10, pady=10)
district_var = tk.StringVar()
district_entry = ttk.Entry(window, textvariable=district_var)
district_entry.grid(row=1, column=1, padx=10, pady=10)

market_label = ttk.Label(window, text="Market:")
market_label.grid(row=2, column=0, padx=10, pady=10)
market_var = tk.StringVar()
market_entry = ttk.Entry(window, textvariable=market_var)
market_entry.grid(row=2, column=1, padx=10, pady=10)

commodity_label = ttk.Label(window, text="Commodity:")
commodity_label.grid(row=3, column=0, padx=10, pady=10)
commodity_var = tk.StringVar()
commodity_entry = ttk.Entry(window, textvariable=commodity_var)
commodity_entry.grid(row=3, column=1, padx=10, pady=10)

variety_label = ttk.Label(window, text="Variety:")
variety_label.grid(row=4, column=0, padx=10, pady=10)
variety_var = tk.StringVar()
variety_entry = ttk.Entry(window, textvariable=variety_var)
variety_entry.grid(row=4, column=1, padx=10, pady=10)

arrival_date_label = ttk.Label(window, text="Arrival Date:")
arrival_date_label.grid(row=5, column=0, padx=10, pady=10)
arrival_date_var = tk.StringVar()
arrival_date_entry = ttk.Entry(window, textvariable=arrival_date_var)
arrival_date_entry.grid(row=5, column=1, padx=10, pady=10)

# Create and place a button to trigger the prediction
predict_button = ttk.Button(window, text="Predict Prices", command=predict_prices)
predict_button.grid(row=6, column=0, columnspan=2, pady=10)

# Create and place labels to display the output
min_price_label = ttk.Label(window, text="Min Price: ")
min_price_label.grid(row=7, column=0, columnspan=2)

max_price_label = ttk.Label(window, text="Max Price: ")
max_price_label.grid(row=8, column=0, columnspan=2)

modal_price_label = ttk.Label(window, text="Modal Price: ")
modal_price_label.grid(row=9, column=0, columnspan=2)

# Run the Tkinter event loop
window.mainloop()
