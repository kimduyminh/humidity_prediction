import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
import pickle
# Import data
data = pd.read_csv(r"weather.csv")

# Setting variables and target variable
y = data.humidi
x = data.drop('humidi', axis=1)
x_train_full, x_test_full, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Drop cols with too much difference value
cate_cols = [cname for cname in x_train_full.columns if x_train_full[cname].nunique() < 10 and x_train_full[cname].dtype == "object"]
nume_cols = [cname for cname in x_train_full.columns if x_train_full[cname].dtype in ['int64', 'float64']]
cols = cate_cols + nume_cols
x_train = x_train_full[cols]
x_test = x_test_full[cols]

# Preprocessing
# Preprocessing for numerical data
#numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
# = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='most_frequent')),
#    ('onehot', OneHotEncoder(handle_unknown='ignore'))
#])

# Bundle preprocessing for numerical and categorical data
#preprocessor = ColumnTransformer(
#    transformers=[
#        ('num', numerical_transformer, nume_cols),
#        ('cat', categorical_transformer, cate_cols)
#    ])

# Define base models
estimators = [
    ('lr', RidgeCV()),
   ('svr', LinearSVR(dual='auto',random_state=42))
]

# Define the final model pipeline
final_model = RandomForestRegressor(n_estimators=2000, random_state=42)

# Create the stacking regressor
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=final_model
)

# Fit the stacking regressor
reg.fit(x_train, y_train)
filename = "decision_tree.pickle"
pickle.dump(reg, open(filename, "wb"))

# Make predictions
prediction = reg.predict(x_test)

# Calculate error:
print("MSR: "+str(mean_squared_error(y_test, prediction)))
print("MAE: "+str(mean_absolute_error(y_test, prediction)))
print("R2: "+str(r2_score(y_test, prediction)))
