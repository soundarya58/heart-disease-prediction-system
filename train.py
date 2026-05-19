import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# LOAD DATASET
data = pd.read_csv("dataset/heart.csv")

# INPUT FEATURES
X = data.drop("HeartDisease", axis=1)

# OUTPUT COLUMN
y = data["HeartDisease"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# SAVE MODEL
pickle.dump(model, open("heart_model.pkl", "wb"))

print("Model trained successfully")