#import required library
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle
#import files and handling datas
data=pd.read_csv("weather.csv")

#setting variables and target variable
y=data.humidi
x=data.drop('humidi', axis=1)
#import files and handling datas
data=pd.read_csv("weather.csv")
encoded_data=pd.get_dummies(data,columns=["province","wind_d","date"])
scaler = StandardScaler()
x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#drop cols with too much difference value
cate_cols=[cname for cname in x_train_full.columns if x_train_full[cname].nunique()<10 and x_train_full[cname].dtype=="object"]
nume_cols=[cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64','float64']]
cols=cate_cols+nume_cols
x_train=x_train_full#[cols]
x_test=x_test_full#[cols]

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


#model1(not optimized)
model_1=Ridge()
pipeline1=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_1)
                             ])

#train the model
pipeline1.fit(x_train,y_train)
prediction1=pipeline1.predict(x_test)
print("Unoptimized Model Mean Error: "+str(mean_absolute_error(y_test,prediction1)))

#model2(Optimizing based on finding the best alpha)
alphas=np.logspace(-4, 2, 100)
mea=[]
progress=1
for i in alphas:
    model_2=Ridge(alpha=i)
    pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model_2)
                                ])
    pipeline2.fit(x_train,y_train)
    mea.append(mean_absolute_error(y_test,pipeline2.predict(x_test)))
    print(str(progress)+"%")
    progress+=1
optimal_alpha=alphas[mea.index(min(mea))]
print("Optimal alpha "+str(optimal_alpha))
model_3=Ridge(alpha=optimal_alpha)
pipeline3=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_3)
                             ])
pipeline3.fit(x_train,y_train)
filename = "ridge.pickle"
pickle.dump(pipeline3, open(filename, "wb"))
prediction=pipeline3.predict(x_test)
print("MSR: "+str(mean_squared_error(y_test, prediction)))
print("MAE: "+str(mean_absolute_error(y_test, prediction)))
print("R2: "+str(r2_score(y_test, prediction)))

#side info: the selection of alpha affect directly and hugely to the accuracy of the prediction,
# so the value of alpha might need to be examine more to decrease the mean error value
