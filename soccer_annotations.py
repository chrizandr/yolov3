"""Read and scale annotations."""
import numpy as np
import os
import pdb
from scipy.io import loadmat
import xml.etree.ElementTree as ET


def get_scaled_annotations_ball(annotation_dir, new_size=(416, 416)):
    """Read and scale annotations based on new image size."""
    files = os.listdir(annotation_dir)
    annotations = dict()
    for f in files:
        try:
            file = ET.parse(os.path.join(annotation_dir, f))
            root = file.getroot()
        except:
            pdb.set_trace()
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

        annotations[f.strip(".xml") + ".png"] = np.array([xmin, ymin, xmax, ymax]).reshape(1, 4)

    return annotations


def get_scaled_annotations_person(matfile, new_size=(416, 416)):
    """Scale annotations based on new image size."""
    mat = loadmat(matfile)
    annotations = mat["annot"][0]
    newannot = dict()
    height, width = 720, 1280
    new_h, new_w = new_size
    new_h, new_w = float(new_h), float(new_w)

    for annot in annotations:
        name = annot[1][0].encode("utf-8")
        bbox = annot[0].astype(np.int)
        bbox[:, [0, 2]] = bbox[:, [0, 2]] / (width/new_w)
        bbox[:, [1, 3]] = bbox[:, [1, 3]] / (height/new_h)
        newannot[name.decode()] = bbox

    return newannot


if __name__ == "__main__":
    # filename = "/home/chris/sports/soccer_ball_data/annotations/scene00741.xml"
    # get_scaled_annotation_ball(filename, (416, 416))
    matfile = "/home/chris/sports/SoccerPlayerDetection_bmvc17_v1/annotation_2.mat"
    get_scaled_annotations_person(matfile, (416, 416))
