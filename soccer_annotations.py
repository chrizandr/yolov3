import xml.etree.ElementTree as ET
from skimage.transform import resize
import numpy as np
import pdb


def get_scaled_annotation(filename, new_size):
    file = ET.parse(filename)
    root = file.getroot()
    annotation = root.findall("object")[0]
    bbox = annotation.findall("bndbox")[0]
    xmin = int(bbox.findall("xmin")[0].text)
    ymin = int(bbox.findall("ymin")[0].text)
    xmax = int(bbox.findall("xmax")[0].text)
    ymax = int(bbox.findall("ymax")[0].text)

    size = root.findall("size")[0]
    width = int(size.findall("width")[0].text)
    height = int(size.findall("height")[0].text)

    new_h, new_w = new_size
    new_h, new_w = float(new_h), float(new_w)
    ymin = int(ymin/(height/new_h))
    ymax = int(ymax/(height/new_h))
    xmin = int(xmin/(width/new_w))
    xmax = int(xmax/(width/new_w))
    pdb.set_trace()


if __name__ == "__main__":
    filename = "/home/chris/sports/soccer_ball_data/annotations/scene00601.xml"
    get_scaled_annotation(filename, (416, 416))
