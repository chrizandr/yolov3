"""Read and scale annotations."""
import numpy as np
import os
import pdb
from scipy.io import loadmat
import xml.etree.ElementTree as ET
from scipy.io import loadmat


def format_annotations(file, outfolder, new_size=(1024, 1024), prefix="D2_",):
    """Format annotation and write to file."""
    width = 1280
    height = 720

    new_h, new_w = new_size
    new_h, new_w = float(new_h), float(new_w)

    mat = loadmat(file)['annot'][0]
    for annot, fname in mat:
        fname = fname[0]
        print(fname)
        f = open(os.path.join(outfolder, prefix + fname.replace(".jpg", ".txt").replace(".png", ".txt")), "w")
        # f = open("sample2.txt", "w")
        for p in annot:
            xmin, ymin, xmax, ymax = p
            ymin = int(ymin/(height/new_h))
            ymax = int(ymax/(height/new_h))
            xmin = int(xmin/(width/new_w))
            xmax = int(xmax/(width/new_w))
            c_x, c_y = int((xmin + xmax) / 2) / 1024.0, int((ymin + ymax) / 2) / 1024.0
            b_w, b_h = abs(xmax - xmin) / 1024.0, abs(ymax - ymin) / 1024.0
            out = ["0"] + [str(x) for x in [c_x, c_y, b_w, b_h]]
            out = " ".join(out) + "\n"
            f.write(out)
        f.close()


def get_scaled_annotations_PVOC(annotation_dir, new_size=None):
    """Read and scale annotations based on new image size."""
    files = os.listdir(annotation_dir)
    annotations = dict()
    for f in files:
        try:
            file = ET.parse(os.path.join(annotation_dir, f))
            root = file.getroot()
        except Exception:
            pdb.set_trace()

        as_ = root.findall("object")
        size = root.findall("size")[0]
        width = int(size.findall("width")[0].text)
        height = int(size.findall("height")[0].text)

        for annotation in as_:
            bbox = annotation.findall("bndbox")[0]
            xmin = float(bbox.findall("xmin")[0].text)
            ymin = float(bbox.findall("ymin")[0].text)
            xmax = float(bbox.findall("xmax")[0].text)
            ymax = float(bbox.findall("ymax")[0].text)

            if new_size:
                new_h, new_w = new_size
                new_h, new_w = float(new_h), float(new_w)
                ymin = float(ymin/(height/new_h))
                ymax = float(ymax/(height/new_h))
                xmin = float(xmin/(width/new_w))
                xmax = float(xmax/(width/new_w))

            name = f.replace(".xml", ".png")
            if name in annotations:
                annotations[name] = np.vstack((annotations[name], [xmin, ymin, xmax, ymax]))
            else:
                annotations[name] = np.array([xmin, ymin, xmax, ymax]).reshape(1, 4)
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


def train_val_split(images):
    files = os.listdir(images)
    indices = np.random.permutation(len(files))
    split_val = int(len(files) * 0.8)
    train = [os.path.join(images, files[i]) for i in indices[0:split_val]]
    val = [os.path.join(images, files[i]) for i in indices[split_val::]]
    f = open("train.txt", "w")
    f.write("\n".join(train))
    f.close()
    f = open("val.txt", "w")
    f.write("\n".join(val))
    f.close()


if __name__ == "__main__":
    # filename = "/home/chrizandr/sports/detection_exp/annotations/"
    # get_scaled_annotations_PVOC(filename, (1024, 1024))
    # matfile = "/home/chris/sports/SoccerPlayerDetection_bmvc17_v1/annotation_2.mat"
    # get_scaled_annotations_person(matfile, (416, 416))
    # format_annotations("/home/chrizandr/sports/SoccerPlayerDetection_bmvc17_v1/annotation_2.mat",
    #                    "/home/chrizandr/sports/train/labels/")
    train_val_split("/home/chrizandr/sports/train/images/")
