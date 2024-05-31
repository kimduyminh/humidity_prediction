import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,f1_score
from sklearn.ensemble import VotingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import pickle
#import data
data=pd.read_csv(r"weather.csv")

#setting variables and target variable
y=data.humidi
x=data.drop('humidi', axis=1)
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

#svr
print("SVR")
lr = LinearRegression()
lr_model=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', lr)
                             ])
lr_model.fit(x_train, y_train)

#random forest
print("RF")
cpu_cores=8
rfr=RandomForestRegressor(n_estimators=2000,random_state=1,n_jobs=cpu_cores)
rfr_model=Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', rfr)
                          ])
rfr_model.fit(x_train, y_train)

#ridge
print("R")
ridge=Ridge(alpha=0.0001)
ridge_model=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', ridge)
                             ])
ridge_model.fit(x_train,y_train)

#voting
ereg = VotingRegressor(estimators=[('ridge', ridge_model), ('rf', rfr_model) , ('lr', lr_model)],weights = [2,3,1])
ereg.fit(x_train, y_train)
filename = "../trained_model/voting_regressor.pickle"
pickle.dump(ereg, open(filename, "wb"))
prediction=ereg.predict(x_test)
print("MSR: "+str(mean_squared_error(y_test, prediction)))
print("MAE: "+str(mean_absolute_error(y_test, prediction)))
print("R2: "+str(r2_score(y_test, prediction)))
