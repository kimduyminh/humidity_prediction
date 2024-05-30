import pandas as pd
import numpy as np
from skfeature.function.similarity_based import fisher_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("weather.csv")

# Define the target and features
y = data.humidi.values  # Convert target to numpy array
x = data.drop('humidi', axis=1).values  # Convert features to numpy array

# Calculate the Fisher score
ranks = fisher_score.fisher_score(x, y)

# Convert to a pandas Series with the correct index
feat_importances = pd.Series(ranks, index=data.drop('humidi', axis=1).columns)

# Plot the data
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
feat_importances.plot(kind='barh')
plt.xlabel("Fisher Score")
plt.ylabel("Features")
plt.title("Feature Importances (Fisher Score)")
plt.show()
