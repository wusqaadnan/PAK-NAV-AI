
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

base = Path(__file__).parent / "RDD2022_India" / "train"

xml_files = list(base.rglob("*.xml"))

if not xml_files:
    print("XML files nahi milin!")

else:
    xml_path = xml_files[0]
    root = ET.parse(xml_path).getroot()

    image_name = root.findtext("filename")

    images = (
        list((base / "images").rglob(Path(image_name).name))
        if image_name else []
    )

    if not images:
        images = list(
            (base / "images").rglob(xml_path.stem + ".*")
        )

    if not images:
        print("Matching image nahi mili:", xml_path.name)

    else:
        img = cv2.imread(str(images[0]))

        if img is None:
            print("Image open nahi hui!")

        else:
            count = 0

            for obj in root.findall("object"):
                box = obj.find("bndbox")

                if box is None:
                    continue

                x1 = int(float(box.findtext("xmin")))
                y1 = int(float(box.findtext("ymin")))
                x2 = int(float(box.findtext("xmax")))
                y2 = int(float(box.findtext("ymax")))

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                count += 1

            print("Annotated potholes:", count)

            # Save image with red bounding boxes
            output_path = Path(__file__).parent / "pothole_detected.jpg"

            cv2.imwrite(str(output_path), img)

            print("Image saved:", output_path)

            # Show image
            cv2.imshow("Pothole Bounding Boxes", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
