#import required libraries

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import os
from timeit import default_timer as timer

#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])

#setting variables and target variable
y=encoded_data.humidi
x=encoded_data.drop('humidi', axis=1)
x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#drop cols with too much difference value
cate_cols=[cname for cname in x_train_full.columns if x_train_full[cname].nunique()<10 and x_train_full[cname].dtype=="object"]
nume_cols=[cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64','float64']]
cols=cate_cols+nume_cols
x_train=x_train_full[cols]
x_test=x_test_full[cols]

#preprocessing
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, nume_cols),
        ('cat', categorical_transformer, cate_cols)
    ])

#model1 (not optimized)
cpu_cores=10
start = timer()
model_1=RandomForestRegressor(n_jobs=cpu_cores)
pipeline1=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_1)
                             ])
pipeline1.fit(x_train,y_train)
prediction1=pipeline1.predict(x_test)
print("Unoptimized model mean error: ")
print(mean_absolute_error(y_test,prediction1))

end =timer()
print(end-start)

#model 2 (Optimized based on number of tree)
n_list=[2000,5000,10000]
#mean calculator function:
def get_mae(n,traX,reaX,traY,reaY):
    model_2=RandomForestRegressor(n_estimators=n,random_state=1,n_jobs=cpu_cores)
    pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model_2)
                                ])
    pipeline2.fit(traX,traY)
    predict_val=pipeline2.predict(reaX)
    mae=mean_absolute_error(reaY,predict_val)
    return mae

#finding best tree number to minimize mean error
mae_data=[]
for i in n_list:
    start = timer()
    print("\n")
    print("Trying tree numbers: "+str(i))
    a=get_mae(i,x_train,x_test,y_train,y_test)
    mae_data.append(a)
    print("Mae: "+str(a))
    end = timer()
    print(str(end - start)+"s")
best_tree_size = n_list[mae_data.index(min(mae_data))]
print("BEST NUMBER OF TREES")
print(best_tree_size)

optimized_model=RandomForestRegressor(n_estimators=best_tree_size,random_state=1,n_jobs=cpu_cores)
optimized_model.fit(x_train,y_train)
predition_optimized=optimized_model.predict(x_test)
print("Optimized model mean error: "+mean_absolute_error(y_test,predition_optimized))
