import tkinter as tk
from tkinter import ttk
import requests
from tabulate import tabulate
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
# Load crop data and train the model
data = pd.read_csv("D:\DESIGN PROJECT\ds1.csv", encoding='ISO-8859-1')
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

# Function to predict temperature and humidity requirements for a crop
def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_index = data[data['Crop'].str.lower() == crop_name].index[0]
    predicted_temperature = model.predict(X.iloc[crop_index].values.reshape(1, -1))
    humidity_required = data.iloc[crop_index]['Humidity Required (%)']
    return humidity_required, predicted_temperature[0]

# Function to get pest warnings for a crop
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}

# Read data from the CSV file and store it in dictionaries
with open("D:\DESIGN PROJECT\ds2.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(row) >= 2:
            crop = row[0].strip().lower()
            pest = row[1].strip()
            crop_pest_data[crop] = pest
        if len(row) >= 3:
            crop = row[0].strip().lower()
            planting_time = row[5].strip()
            planting_time_info[crop] = planting_time
            growth_stage = row[6].strip()
            growth_stage_info[crop] = growth_stage

# Function to predict pest warnings for a given crop
def predict_pest_warnings(crop_name):
    crop_name = crop_name.lower()
    specified_crops = [crop_name]

    pest_warnings = []

    for crop in specified_crops:
        if crop in crop_pest_data:
            pests = crop_pest_data[crop].split(', ')
            warning_message = f"Beware of pests like {', '.join(pests)} for {crop.capitalize()}."

            if crop in planting_time_info:
                planting_time = planting_time_info[crop]
                warning_message += f"\nPlanting Time: {planting_time}"

            if crop in growth_stage_info:
                growth_stage = growth_stage_info[crop]
                warning_message += f"\nGrowth Stages of Plant: {growth_stage}"

            pest_warnings.append(warning_message)

    return '\n'.join(pest_warnings)  

# Function to fetch and display weather forecast
def fetch_and_display_weather():
    village_name = village_entry.get()

    # Use a geocoding service to convert the village name to coordinates
    geocoding_api_key = '80843f03ed6b4945a45f1bd8c51e5c2f'  
    geocoding_url = f'https://api.opencagedata.com/geocode/v1/json?q={village_name}&key={geocoding_api_key}'

    geocoding_response = requests.get(geocoding_url)
    if geocoding_response.status_code == 200:
        geocoding_data = geocoding_response.json()
        if geocoding_data['results']:
            # Extract latitude and longitude from geocoding response
            latitude = geocoding_data['results'][0]['geometry']['lat']
            longitude = geocoding_data['results'][0]['geometry']['lng']
            api_key = 'b53305cd6b960c1984aed0acaf76aa2e'
            weather_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&units=metric&cnt=40&appid={api_key}'
            weather_response = requests.get(weather_url)

            if weather_response.status_code == 200:
                weather_data = weather_response.json()

                daily_weather_data = defaultdict(list)

                for forecast in weather_data['list']:
                    date = forecast['dt_txt'].split()[0]  
                    daily_weather_data[date].append(forecast)

                forecast_text.delete('1.0', tk.END)  

                for date, forecasts in daily_weather_data.items():
                    weather_updates = []

                    for forecast in forecasts:
                        time = forecast['dt_txt'].split()[1]  
                        temperature = forecast['main']['temp']
                        feels_like = forecast['main']['feels_like']
                        weather_description = forecast['weather'][0]['description']
                        wind_speed = forecast['wind']['speed']
                        wind_direction = forecast['wind']['deg']
                        pressure = forecast['main']['pressure']
                        humidity = forecast['main']['humidity']

                        weather_updates.append([time, temperature, feels_like, weather_description, wind_speed, wind_direction, pressure, humidity])

                    headers = ['Time', 'Temperature (°C)', 'Feels Like (°C)', 'Description', 'Wind Speed (m/s)', 'Wind Direction (°)', 'Pressure (hPa)', 'Humidity (%)']
                    weather_table = tabulate(weather_updates, headers=headers, tablefmt='pretty')

                    forecast_text.insert(tk.END, f"Weather Forecast for {date}:\n")
                    forecast_text.insert(tk.END, weather_table)
                    forecast_text.insert(tk.END, '\n\n')
            else:
                forecast_text.delete('1.0', tk.END) 
                forecast_text.insert(tk.END, f'Error: Unable to fetch weather data. Status code {weather_response.status_code}')
        else:
            forecast_text.delete('1.0', tk.END) 
            forecast_text.insert(tk.END, f'Error: No geocoding results found for the village name.')
    else:
        forecast_text.delete('1.0', tk.END)  
        forecast_text.insert(tk.END, f'Error: Unable to perform geocoding. Status code {geocoding_response.status_code}')


# Create the main tkinter window
root = tk.Tk()
root.title("Farmer-Desk")

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

crop_frame = ttk.Frame(notebook)
weather_frame = ttk.Frame(notebook)
recommendation_frame = ttk.Frame(notebook)

notebook.add(crop_frame, text="Crop Requirements & Pest Warnings")
notebook.add(weather_frame, text="Weather Forecast")
notebook.add(recommendation_frame, text="Crop Recommendation")  # Add the Crop Recommendation tab

# --- Crop Requirements & Pest Warnings Tab ---
crop_label = ttk.Label(crop_frame, text="Enter the name of the crop:")
crop_label.pack(pady=5)
crop_entry = ttk.Entry(crop_frame)
crop_entry.pack(pady=5)

# Function to display crop requirements, pest warnings
def display_crop_info():
    crop_name = crop_entry.get()
    humidity, temperature = predict_requirements(crop_name)

    crop_result_text.config(state=tk.NORMAL)
    crop_result_text.delete("1.0", tk.END)

    pest_warning = predict_pest_warnings(crop_name)
    if pest_warning:
        crop_result_text.insert(tk.END, f'Beware of pests like {pest_warning}\n')

    crop_result_text.insert(tk.END, f'Predicted Humidity Required (%): {humidity:.2f}\n')
    crop_result_text.insert(tk.END, f'Predicted Temperature Required (°F): {temperature:.2f}\n')

    crop_result_text.config(state=tk.DISABLED)

calculate_crop_button = ttk.Button(crop_frame, text="Calculate", command=display_crop_info)
calculate_crop_button.pack(pady=10)

crop_result_text = tk.Text(crop_frame, wrap=tk.WORD, height=100, width=140)
crop_result_text.pack()

# --- Crop Recommendation Tab ---

# Label and Entry for input features
recommend_label = ttk.Label(recommendation_frame, text="Crop Recommendation:")
recommend_label.pack(pady=10)

# Create input fields and labels
N_label = ttk.Label(recommendation_frame, text="Nitrogen (N):")
N_label.pack()
N_entry = ttk.Entry(recommendation_frame)
N_entry.pack()

P_label = ttk.Label(recommendation_frame, text="Phosphorous (P):")
P_label.pack()
P_entry = ttk.Entry(recommendation_frame)
P_entry.pack()

K_label = ttk.Label(recommendation_frame, text="Potassium (K):")
K_label.pack()
K_entry = ttk.Entry(recommendation_frame)
K_entry.pack()

temperature_label = ttk.Label(recommendation_frame, text="Temperature (°C):")
temperature_label.pack()
temperature_entry = ttk.Entry(recommendation_frame)
temperature_entry.pack()

humidity_label = ttk.Label(recommendation_frame, text="Humidity (%):")
humidity_label.pack()
humidity_entry = ttk.Entry(recommendation_frame)
humidity_entry.pack()

ph_label = ttk.Label(recommendation_frame, text="pH Value:")
ph_label.pack()
ph_entry = ttk.Entry(recommendation_frame)
ph_entry.pack()

rainfall_label = ttk.Label(recommendation_frame, text="Rainfall (mm):")
rainfall_label.pack()
rainfall_entry = ttk.Entry(recommendation_frame)
rainfall_entry.pack()

# Function to recommend a crop using the trained model
def recommend_crop():
    crop_recommendation_data = pd.read_csv("D:\DESIGN PROJECT\Crop_recommendation.csv")
    X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = crop_recommendation_data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    N = float(N_entry.get())
    P = float(P_entry.get())
    K = float(K_entry.get())
    temperature = float(temperature_entry.get())
    humidity = float(humidity_entry.get())
    ph = float(ph_entry.get())
    rainfall = float(rainfall_entry.get())

    user_input = [[N, P, K, temperature, humidity, ph, rainfall]]
    recommended_crop = model.predict(user_input)[0]

    recommendation_result.config(text=f"Recommended Crop: {recommended_crop}")

recommend_button = ttk.Button(recommendation_frame, text="Recommend Crop", command=recommend_crop)
recommend_button.pack()

recommendation_result = ttk.Label(recommendation_frame, text="")
recommendation_result.pack()

# --- Weather Forecast Tab ---

village_label = ttk.Label(weather_frame, text="Enter the name of the village:")
village_label.pack(pady=5)
village_entry = ttk.Entry(weather_frame)
village_entry.pack(pady=5)

fetch_weather_button = ttk.Button(weather_frame, text="Fetch Weather Forecast", command=fetch_and_display_weather)
fetch_weather_button.pack(pady=10)

forecast_text = tk.Text(weather_frame, wrap=tk.WORD, height=100, width=140)
forecast_text.pack()

# Start the tkinter main loop
root.mainloop()
