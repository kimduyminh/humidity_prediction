#import required library
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle

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

#model1 (Not optimized)
model=Lasso()
pipeline=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

#train the model
pipeline.fit(x_train,y_train)
prediction1=pipeline.predict(x_test)
print([float(i) for i in model.coef_])
print(model.alpha)
print("Mae "+str(mean_absolute_error(y_test,prediction1)))

#model2 (Trying optimized arguments a bit by figuring what is the best alpha)
lasso_cv=LassoCV(max_iter=100,cv=5)
pipeline1=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', lasso_cv)
                             ])
pipeline1.fit(x_train,y_train)
optimal_alpha=lasso_cv.alpha_
print("Optimal alpha: "+str(optimal_alpha))
model_2=Lasso(alpha=lasso_cv.alpha_)
pipeline3=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model_2)
                             ])
pipeline3.fit(x_train, y_train)
filename = "../trained_model/lasso.pickle"
pickle.dump(pipeline3, open(filename, "wb"))
prediction=pipeline3.predict(x_test)
print(model_2.coef_)
print("MSR: "+str(mean_squared_error(y_test, prediction)))
print("MAE: "+str(mean_absolute_error(y_test, prediction)))
print("R2: "+str(r2_score(y_test, prediction)))

