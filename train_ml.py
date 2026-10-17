
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

folder = Path(__file__).parent

# Read dataset
data = pd.read_csv(folder / "pothole_ml_data.csv")

print("Total records:", len(data))

# Extract numerical features from images
import cv2
import numpy as np

features = []
targets = []

image_folder = folder / "RDD2022_India" / "train" / "images"

for _, row in data.iterrows():
    image_path = image_folder / row["image"]
    img = cv2.imread(str(image_path))

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    features.append([
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    ])

    targets.append(float(row["pothole_count"]))

print("Images loaded:", len(features))

if len(features) < 10:
    raise ValueError("Training ke liye images nahi milin.")

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

joblib.dump(model, folder / "initial_ml_model.pkl")

print("ML MODEL TRAINED SUCCESSFULLY!")
print("Model saved: initial_ml_model.pkl")
