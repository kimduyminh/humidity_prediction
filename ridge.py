#import required library
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])
scaler = StandardScaler()
encoded_data_scaled = pd.DataFrame(scaler.fit_transform(encoded_data), columns=encoded_data.columns)

#setting variables and target variable
y=encoded_data.humidi
x=encoded_data.drop('humidi', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Check for NaN or infinite values in your data
print("Any NaN in data: ", pd.isna(encoded_data).any().any())
print("Any Inf in data: ", np.isinf(encoded_data).any().any())




#model1(not optimized)
model_1=Ridge()

#train the model
model_1.fit(x_train,y_train)
prediction1=model_1.predict(x_test)
print("mae "+str(mean_absolute_error(y_test,prediction1)))

#model2(Optimizing based on finding the best alpha)
alphas=np.logspace(-4, 2, 100)
mea=[]
progress=1
for i in alphas:
    model_cv=RidgeCV(alphas=i,cv=5)
    model_cv.fit(x_train,y_train)
    mea.append(mean_absolute_error(y_test,model_cv.predict(x_test)))
    print(str(progress)+"%")
    progress+=1
optimal_alpha=min(mea)
model_2=Ridge(alpha=optimal_alpha)
model_2.fit(x_train,y_train)
prediction2=model_2.predict(x_test)
print("mae "+str(mean_absolute_error(y_train,prediction2)))

#side info: the selection of alpha affect directly and hugely to the accuracy of the prediction,
# so the value of alpha might need to be examine more to decrease the mean error value
