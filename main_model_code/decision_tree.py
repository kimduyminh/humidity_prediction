#import required packages

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pickle
#import data
data=pd.read_csv("weather.csv")

#setting variables and target variable
y=data.humidi
x=data.drop('humidi', axis=1)
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


#model 1 (Not optimized)
model1=DecisionTreeRegressor(random_state=1)
pipeline1=Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model1)
                             ])
pipeline1.fit(x_train,y_train)
prediction1=pipeline1.predict(x_test)
print("Nonoptimized model mean error: ")
print(mean_absolute_error(y_test,prediction1))

#model 2 (Optimized based on tree max_leaf_nodes)
#mean calculator function:
def get_mae(max_leaf_nodes,traX,reaX,traY,reaY):
    model_2=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=1)
    pipeline2 = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model_2)
                                ])
    pipeline2.fit(traX,traY)
    predict_val=pipeline2.predict(reaX)
    mae=mean_absolute_error(reaY,predict_val)
    return mae

#finding best tree size to minimize mean error

sample_max_leaf_nodes=[100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000]
mae_data=[]
for i in sample_max_leaf_nodes:
    print("Trying leaf size: "+str(i))
    a=get_mae(i,x_train,x_test,y_train,y_test)
    mae_data.append(a)
    print(str(a))
best_tree_size = sample_max_leaf_nodes[mae_data.index(min(mae_data))]
print("BEST TREE SIZE")
print(best_tree_size)

'''best_tree_size=100000'''
#after several test, we found out the max_leaf_nodes that given a good mean error is 100000
#best_tree_size=100000
final_model=DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=1)
pipeline3 = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', final_model)
                                ])
pipeline3.fit(x_train,y_train)
filename = "../trained_model/decision_tree.pickle"
pickle.dump(pipeline3, open(filename, "wb"))
print("Optimized model mean error: ")
print(x_test.head(3))
prediction=pipeline3.predict(x_test)
print("MSR: "+str(mean_squared_error(y_test, prediction)))
print("MAE: "+str(mean_absolute_error(y_test, prediction)))
print("R2: "+str(r2_score(y_test, prediction)))


