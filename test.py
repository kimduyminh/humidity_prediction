import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#import data
data=pd.read_csv(r"main_model/weather.csv")
print(data.iloc[:, 0].unique())
print(data.iloc[:, 4].unique())
print(data.columns)
print(data.head(2))
print("Enter these info respectively: ")
print("Province, max temp, min temp, wind speed, wind direction, rainfall(mm),cloud percentage, air pressure(hPa)")
info=input().split()
print(info)
#setting variables and target variable
#y=data.humidi
#x=data.drop('humidi', axis=1)
#x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
#decision_tree=pickle.load(open('decision_tree.pickle','rb'))
#prediction=decision_tree.predict(x_test_full)
#print("MSR: "+str(mean_squared_error(y_test, prediction)))
#print("MAE: "+str(mean_absolute_error(y_test, prediction)))
#print("R2: "+str(r2_score(y_test, prediction)))
