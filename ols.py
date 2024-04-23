#import required libs
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])

#setting variables and target variable
y=encoded_data.humidi
x=encoded_data.drop('humidi', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#create model
model=linear_model.LinearRegression()

#train model
model.fit(x_train,y_train)

#print the model coefficients
coef=model.coef_
coef_int=[float(i) for i in coef]
print(coef)
#calculate the mean error
prediction=model.predict(x_test)
print("Mae: "+str(mean_absolute_error(y_test,prediction)))