import xml.etree.ElementTree as ET
import glob
import os
import cv2
from tqdm import tqdm

def read_voc_xml(xmlfile: str) -> dict:
    root = ET.parse(xmlfile).getroot()
    boxes = {"filename": root.find("filename").text,
             "objects": []}
    
    for box in root.iter('object'):
        bb = box.find('bndbox')
        obj = {
            "name": box.find('name').text,
            "xmin": int(bb.find("xmin").text),
            "ymin": int(bb.find("ymin").text),
            "xmax": int(bb.find("xmax").text),
            "ymax": int(bb.find("ymax").text),
        }
        boxes["objects"].append(obj)

    return boxes


def create_annotation_haar_cascade_format(dir):
    xml_files = glob.glob(os.path.join(dir, '*.xml'))
    xml_files = sorted(xml_files)
    lines = []

    for fielname in tqdm(xml_files):
        boxes = read_voc_xml(fielname)
        objects = boxes['objects']
        line = f"{fielname.split('./')[-1].replace('.xml', '.jpg')} {len(objects)} "

        for object in objects:
            xmin = object['xmin']
            ymin = object['ymin']
            xmax = object['xmax']
            ymax = object['ymax']

            if xmax > 640:
                xmax = 640
            if ymax > 640:
                ymax = 640

            temp = f'{xmin} {ymin} {xmax - xmin} {ymax - ymin} '
            line += temp

        line = line[:-1]
        lines.append(line)

    
    with open("positive_1000.dat", 'w') as f:
        f.write("\n".join(lines))
        

# create_annotation_haar_cascade_format("./images/positive_1000")


def plot_true_boxes(filename):
    image = cv2.imread(filename)
    boxes = read_voc_xml(filename.replace('.jpg', '.xml'))
    objects = boxes['objects']

    for object in objects:
        xmin = object['xmin']
        ymin = object['ymin']
        xmax = object['xmax']
        ymax = object['ymax']
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow("Image with True Boxes", image)
    cv2.waitKey(0)

# plot_true_boxes("images/positive_1000/MVI_20011_img00120_jpg.rf.423c07d7cec9b897b7b9d202a169c065.jpg")



def list_txt_files(input_dir, output_file):
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    with open(output_file, 'w') as out:
        for filename in txt_files:
            full_path = os.path.join(input_dir, filename)
            out.write(full_path + '\n')

    print(f"Listed {len(txt_files)} .txt files to {output_file}")

input_directory = 'images/negative_2000'
output_file_path = 'negative_2000.dat'

# list_txt_files(input_directory, output_file_path)