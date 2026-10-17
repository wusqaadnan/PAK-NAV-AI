
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path(__file__).parent / "RDD2022_India" / "train"
output = Path(__file__).parent / "pothole_results"
output.mkdir(exist_ok=True)

xml_files = list(base.rglob("*.xml"))
saved = 0

for xml_path in xml_files:
    root = ET.parse(xml_path).getroot()

    image_name = root.findtext("filename")
    if image_name:
        image_name = Path(image_name.replace("\\", "/")).name
    else:
        image_name = xml_path.stem + ".jpg"

    image_path = base / "images" / image_name

    if not image_path.exists():
        matches = list((base / "images").rglob(xml_path.stem + ".*"))
        if not matches:
            continue
        image_path = matches[0]

    img = cv2.imread(str(image_path))
    if img is None:
        continue

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue

        x1 = int(float(box.findtext("xmin")))
        y1 = int(float(box.findtext("ymin")))
        x2 = int(float(box.findtext("xmax")))
        y2 = int(float(box.findtext("ymax")))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imwrite(str(output / image_path.name), img)
    saved += 1

print("Total processed images:", saved)
print("Saved in:", output)
