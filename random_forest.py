#import required libraries

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])

#setting variables and target variable
y=encoded_data.humidi
x=encoded_data.drop('humidi', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#model1 (not optimized)
model1=RandomForestRegressor()
model1.fit(x_train,y_train)
prediction1=model1.predict(x_test)
print("First model mean error: ")
print(mean_absolute_error(y_test,prediction1))

#model 2 (Optimized based on tree max_leaf_nodes)
n_list=[5,10,20,50,100,200,500]

#mean calculator function:
def get_mae(n,traX,reaX,traY,reaY):
    model_2=RandomForestRegressor(n_estimators=n,random_state=1)
    model_2.fit(traX,traY)
    predict_val=model_2.predict(reaX)
    mae=mean_absolute_error(reaY,predict_val)
    return mae

#finding best tree number to minimize mean error

mae_data=[]
for i in n_list:
    print("Trying leaf size: "+str(i))
    mae_data.append(get_mae(i,x_train,x_test,y_train,y_test))
best_tree_size = n_list[mae_data.index(min(mae_data))]
print("BEST NUMBER OF TREES") #100000
print(best_tree_size)

