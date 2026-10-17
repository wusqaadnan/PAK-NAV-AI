
import cv2
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

# Load trained ML model
model = joblib.load(folder / "initial_ml_model.pkl")

# Load image information
data = pd.read_csv(folder / "pothole_ml_data.csv")
image_folder = folder / "RDD2022_India" / "train" / "images"

# Select a test image
row = data.iloc[0]
image_path = image_folder / row["image"]

img = cv2.imread(str(image_path))

if img is None:
    print("Image nahi mili:", image_path)
else:
    # Same features used during training
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    features = [[
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    ]]

    prediction = model.predict(features)[0]

    print("Image:", row["image"])
    print("Actual pothole count:", row["pothole_count"])
    print("Predicted pothole count:", round(float(prediction), 2))
    print("ML MODEL TEST COMPLETED!")

    # Save result
    result_file = folder / "ml_test_result.txt"

    with open(result_file, "w") as f:
        f.write(f"Image: {row['image']}\n")
        f.write(f"Actual count: {row['pothole_count']}\n")
        f.write(f"Predicted count: {prediction:.2f}\n")

    print("Result saved:", result_file)
