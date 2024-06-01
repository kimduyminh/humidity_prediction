# Rainfall Prediction <a name="rainfall-prediction"></a>

## Table of Contents
* [Rainfall-Prediction](#rainfall-prediction)
  * [About](#about)
  * [Installation](#installation)
  * [Usage](#usage)


## About <a name="about"></a>

The Rainfall Prediction app is a machine learning-based application designed to predict the amount of rainfall based on historical weather data. This tool can be useful for farmers, meteorologists, and anyone interested in weather patterns.


## Installation <a name="installation"></a>

To use this package, follow these steps:

1. Clone the repository: git clone https://github.com/kimduyminh/humidity_prediction
2. Navigate to the project directory: cd Rainfall-Prediction
3. Install dependencies: pip install -r requirements.txt
4. Download the pretrained model which is not included because of model size, or you can just run the model inside main_model_code
    https://gamecrack-my.sharepoint.com/:f:/g/personal/game_gamecrack_onmicrosoft_com/Eq8XprAezWdOvXIvsCq_dWsB3dSOiZF60HxCl6PqchRW6A?e=aV6hfM
## Usage <a name="usage"></a>

To use the Rainfall Prediction app, follow these steps:
### 1. Load the Application
First, ensure that you have all the required dependencies installed. Then, run the application by executing the following command in your terminal:
python app.py
This will start the Tkinter GUI for the Rainfall Prediction app.

### 2. Enter Input Data
In the GUI, you will see several fields where you can input weather data:

+Province: Enter the name of the province.

+Max Temperature: Enter the maximum temperature (in degrees Celsius).

+Min Temperature: Enter the minimum temperature (in degrees Celsius).

+Wind Speed: Enter the wind speed (in km/h).

+Wind Direction: Enter the wind direction (e.g., N, NE, E, etc.).

+Rainfall: Enter the amount of rainfall (in mm).

+Cloud Cover: Enter the percentage of cloud cover.

+Pressure: Enter the atmospheric pressure (in hPa).

+Date: Enter the date in the format YYYY-MM-DD.

### 3. Make Predictions
After entering the data, you can make predictions using different machine learning models by clicking on the respective buttons:

+Predict Decision Tree: Click to get the prediction from the Decision Tree model.

+Predict Voting Regressor: Click to get the prediction from the Voting Regressor model.

+Predict Stacking Tree: Click to get the prediction from the Stacking Tree model.

+Predict Ridge: Click to get the prediction from the Ridge model.

+Predict Random Forest: Click to get the prediction from the Random Forest model.

+Predict OLS: Click to get the prediction from the OLS (Ordinary Least Squares) model.

+Predict Lasso: Click to get the prediction from the Lasso model.

### 4. View Results
The prediction result will be displayed at the bottom of the GUI under the "Result" label. For example, if you clicked on "Predict Decision Tree," the result will be shown as:

Decision Tree: [Prediction Value]

### Example
Here's an example of how to use the app:

1.Enter the following data:

Province: Hanoi
Max Temperature: 28
Min Temperature: 24
Wind Speed: 4
Wind Direction: SW
Rainfall: 20
Cloud Cover: 80
Pressure: 1003
Date: 2024-5-31

2.Click on the "Predict Decision Tree" button.

3.The result will be displayed as: Decision Tree: 87.0

Repeat the steps for other models if needed.

Feel free to explore different input values and models to see how predictions vary.
## Technologies Used
- Python: The core language used for development.
- Pandas: For data manipulation and analysis.
- Scikit-learn: For building and training the machine learning model.