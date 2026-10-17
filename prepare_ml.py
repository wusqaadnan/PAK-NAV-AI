
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path(__file__).parent / "RDD2022_India" / "train"
output = Path(__file__).parent / "pothole_ml_data.csv"

rows = []

for xml_file in base.rglob("*.xml"):
    root = ET.parse(xml_file).getroot()
    objects = root.findall("object")

    rows.append({
        "image": xml_file.stem + ".jpg",
        "pothole_count": len(objects),
        "label": 1 if len(objects) > 0 else 0
    })

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image", "pothole_count", "label"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("Total records:", len(rows))
print("CSV saved:", output)
