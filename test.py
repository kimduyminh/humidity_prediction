import tkinter as tk
import pickle
import pandas as pd

# Function to load model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to unload model
def unload_model(model):
    del model

# Load models that are lightweight or used frequently at the start
decision_tree = load_model('trained_model/decision_tree.pickle')
ridge = load_model('trained_model/ridge.pickle')
ols = load_model('trained_model/ols.pickle')
lasso = load_model('trained_model/lasso.pickle')

# Function to create a DataFrame from user input
def get_input_data():
    province = province_entry.get()
    max_temp = float(max_entry.get())
    min_temp = float(min_entry.get())
    wind_speed = float(wind_entry.get())
    wind_direction = wind_d_entry.get()
    rainfall = float(rain_entry.get())
    cloud_cover = float(cloud_entry.get())
    pressure = float(pressure_entry.get())
    date = date_entry.get()

    # Create a dictionary from user input
    data = {
        "province": [province],
        "max": [max_temp],
        "min": [min_temp],
        "wind": [wind_speed],
        "wind_d": [wind_direction],
        "rain": [rainfall],
        "cloud": [cloud_cover],
        "pressure": [pressure],
        "date": [date]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

# Functions for each model prediction
def predict_dt():
    df = get_input_data()
    dt_prediction = decision_tree.predict(df)
    result_label.config(text=f"Decision Tree: {dt_prediction[0]}")

def predict_vr():
    df = get_input_data()
    voting_regressor = load_model('trained_model/voting_regressor.pickle')
    vr_prediction = voting_regressor.predict(df)
    unload_model(voting_regressor)
    result_label.config(text=f"Voting Regressor: {vr_prediction[0]}")

def predict_st():
    df = get_input_data()
    stacking_tree = load_model('trained_model/stacking_tree.pickle')
    st_prediction = stacking_tree.predict(df)
    unload_model(stacking_tree)
    result_label.config(text=f"Stacking Tree: {st_prediction[0]}")

def predict_ridge():
    df = get_input_data()
    ridge_prediction = ridge.predict(df)
    result_label.config(text=f"Ridge: {ridge_prediction[0]}")

def predict_rf():
    df = get_input_data()
    random_forest = load_model('trained_model/random_forest.pickle')
    rf_prediction = random_forest.predict(df)
    unload_model(random_forest)
    result_label.config(text=f"Random Forest: {rf_prediction[0]}")

def predict_ols():
    df = get_input_data()
    ols_prediction = ols.predict(df)
    result_label.config(text=f"OLS: {ols_prediction[0]}")

def predict_lasso():
    df = get_input_data()
    lasso_prediction = lasso.predict(df)
    result_label.config(text=f"Lasso: {lasso_prediction[0]}")

# Load models that are lightweight or used frequently at the start
with open('trained_model/decision_tree.pickle', 'rb') as dt_file:
    decision_tree = pickle.load(dt_file)
with open('trained_model/ridge.pickle', 'rb') as ridge_file:
    ridge = pickle.load(ridge_file)
with open('trained_model/ols.pickle', 'rb') as ols_file:
    ols = pickle.load(ols_file)
with open('trained_model/lasso.pickle', 'rb') as lasso_file:
    lasso = pickle.load(lasso_file)

# Create the GUI
root = tk.Tk()
root.title("Weather Model Predictor")

# Create input fields
province_label = tk.Label(root, text="Province:")
province_entry = tk.Entry(root)
max_label = tk.Label(root, text="Max Temperature:")
max_entry = tk.Entry(root)
min_label = tk.Label(root, text="Min Temperature:")
min_entry = tk.Entry(root)
wind_label = tk.Label(root, text="Wind Speed:")
wind_entry = tk.Entry(root)
wind_d_label = tk.Label(root, text="Wind Direction:")
wind_d_entry = tk.Entry(root)
rain_label = tk.Label(root, text="Rainfall:")
rain_entry = tk.Entry(root)
cloud_label = tk.Label(root, text="Cloud Cover:")
cloud_entry = tk.Entry(root)
pressure_label = tk.Label(root, text="Pressure:")
pressure_entry = tk.Entry(root)
date_label = tk.Label(root, text="Date (YYYY-MM-DD):")
date_entry = tk.Entry(root)
# Create prediction buttons for each model
predict_dt_button = tk.Button(root, text="Predict Decision Tree", command=predict_dt)
predict_vr_button = tk.Button(root, text="Predict Voting Regressor", command=predict_vr)
predict_st_button = tk.Button(root, text="Predict Stacking Tree", command=predict_st)
predict_ridge_button = tk.Button(root, text="Predict Ridge", command=predict_ridge)
predict_rf_button = tk.Button(root, text="Predict Random Forest", command=predict_rf)
predict_ols_button = tk.Button(root, text="Predict OLS", command=predict_ols)
predict_lasso_button = tk.Button(root, text="Predict Lasso", command=predict_lasso)

# Create result label
result_label = tk.Label(root, text="")

# Pack widgets
province_label.pack()
province_entry.pack()
max_label.pack()
max_entry.pack()
min_label.pack()
min_entry.pack()
wind_label.pack()
wind_entry.pack()
wind_d_label.pack()
wind_d_entry.pack()
rain_label.pack()
rain_entry.pack()
cloud_label.pack()
cloud_entry.pack()
pressure_label.pack()
pressure_entry.pack()
date_label.pack()
date_entry.pack()

predict_dt_button.pack()
predict_vr_button.pack()
predict_st_button.pack()
predict_ridge_button.pack()
predict_rf_button.pack()
predict_ols_button.pack()
predict_lasso_button.pack()

result_label.pack()

root.mainloop()