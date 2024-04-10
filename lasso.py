<<<<<<< HEAD
#import required library
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
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
print(model.coef_)
print(model.alpha)
print("Mae "+str(mean_absolute_error(y_test,prediction1)))

#model2 (Trying optimized arguments a bit by figuring what is the best alpha)
lasso_cv=LassoCV(max_iter=100,cv=5)
lasso_cv.fit(x_train,y_train)
optimal_alpha=lasso_cv.alpha_
print(optimal_alpha)
model_2=Lasso(alpha=lasso_cv.alpha_)
prediction2=model_2.predict(x_test)
print(model_2.coef_)
print("Mae "+str(mean_absolute_error(y_test,prediction2)))

=======
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
>>>>>>> 2193f69c8866bccf2b6ace78be85fe8edc439cda
