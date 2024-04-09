#import required library
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error

#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])

#setting variables and target variable
y=encoded_data.humidi
x=encoded_data.drop('humidi', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#model1 (Not optimized)
model=Lasso(alpha=0.1)

#train the model
model.fit(x_train,y_train)
prediction1=model.predict(x_test)
print("Mae "+mean_absolute_error(y_test,prediction1))

#model2 (Trying optimized arguments a bit)
model2=Lasso(alpha=0.2)
