# importing required libraries

from xgboost import XGBRegressor
import pandas as pd
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os

#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])

#setting variables and target variable
y=encoded_data.humidi
x=encoded_data.drop('humidi', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#model 1(Not optimized)
start=timer()
model_1=XGBRegressor()
model_1.fit(x_train,y_train)
prediction1=model_1.predict(x_test)
print("Mae: "+str(mean_absolute_error(y_test,prediction1)))
end=timer()
print(str(end-start)+"s")

#model_2 (Optimizing a bit lol)
start=timer()
cpu_cores=os.cpu_count()-2
model_2=XGBRegressor(random_state=0,n_estimators=10000,learning_rate=0.05,n_jobs=cpu_cores)
model_2.set_params(early_stopping_rounds=10)
model_2.fit(x_train,y_train,eval_set=[(x_train,y_train)])
prediction2=model_2.predict(x_test)
print("Mae: "+str(mean_absolute_error(y_test,prediction2)))