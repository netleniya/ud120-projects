import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("../tools/"))

from feature_format import featureFormat


# read in data dictionary, convert to numpy array
with open("../final_project/final_project_dataset.pkl", "rb") as file:
    data_dict = joblib.load(file)
    data_dict.pop("TOTAL", 0)


features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

salary = np.array(data[:, 0]).reshape(-1, 1)
bonus = np.array(data[:, 1]).reshape(-1, 1)


# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    salary, bonus, test_size=0.1, random_state=42
)

# Linear Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print(f"score: {reg.score(X_test, y_test):.3f}")


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()

sns.scatterplot(x=salary[:, 0], y=bonus[:, 0], ax=ax)
sns.lineplot(x=X_test[:, 0], y=y_pred[:, 0], ax=ax, color="r")
sns.despine()

fig.suptitle("Bonus vs Salary")
ax.set_xlabel("Salary")
ax.set_ylabel("Bonus")

plt.show()


# find executives making more than $1e6 in salary and $5e6 in bonuses

greedy = {
    key
    for key, val in data_dict.items()
    if isinstance(val.get("salary"), int)
    and val.get("salary") > 1000000
    and val.get("bonus") > 5000000
}

print(f"\nThe following {len(greedy)} make too much")
for name in greedy:
    print(name)
