"""Evaluate results and find the reasons for errors."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import code
import pdb
from soccer_annotations import get_scaled_annotations_PVOC, get_scaled_annotations_person
from utils import utils


def read_output_file(filename):
    """Read the output files for predictions."""
    annotations = []
    confs = []
    with open(filename, "r") as f:
        for line in f:
            d = line.split()
            cords = [int(float(x)) for x in d[0:4]]
            conf = float(d[-1])
            annotations.append(cords)
            confs.append(conf)
    return np.array(annotations), np.array(confs)


def evaluate(output_dir, data_dir, annotation_dir, size, overlap=0.5, filter_str1="", filter_str2=""):
    """Evaluate output for the ball data."""
    files = os.listdir(data_dir)
    output_files = os.listdir(output_dir)
    output_files = [x for x in output_files if x.endswith(".txt")]
    output_imgs = [os.path.splitext(x)[0] for x in output_files]
    annotations = get_scaled_annotations_PVOC(annotation_dir, size)
    aps = []
    for f in files:
        annotation = annotations[f]
        if filter_str1 not in f or filter_str2 not in f:
            continue
        if f not in output_imgs:
            aps.append(0)
            # print(f, f not in output_imgs)
        else:
            preds, conf = read_output_file(os.path.join(output_dir, f + ".txt"))
            if len(preds) == 0:
                aps.append(0)
            else:
                target_cls = np.zeros(annotation.shape[0])
                pred_cls = np.zeros(preds.shape[0])
                tps = match_annotations(preds, annotation, overlap)
                p, r, ap, f1, _ = utils.ap_per_class(tps, conf, pred_cls, target_cls)
                aps.append(ap[0])
    mean_ap = sum(aps) / len(aps)
    return mean_ap


def map_voc_range(output_dir, data_dir, annotation_dir,  range=[0.5], filter_str1="", filter_str2="", size=(1024, 1024)):
    """Find AP@[.5:.95]."""
    assert len(range) > 0
    maps = []
    for r in range:
        map = evaluate(output_dir, data_dir, annotation_dir, overlap=r,
                       filter_str1=filter_str1, filter_str2=filter_str2,
                       size=size)
        maps.append(map)
    return sum(maps) / len(maps)


def match_annotations(output, annotation, overlap):
    """Match the annotations and output in the image and find accuracy."""
    tps = []
    for o in output:
        # pdb.set_trace()
        dist = np.sum((annotation - o) ** 2, axis=1)
        closest_box = annotation[np.argmin(dist), :]
        iou = bb_intersection_over_union(closest_box, o)
        tps.append(int(iou >= overlap))

    return np.array(tps)


def bb_intersection_over_union(boxA, boxB):
    """Find IOU between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def mark_detection(img_file, output, annotation, output_dir):
    """Mark detection in image."""
    img = cv2.imread(img_file)

    for x in output:
        cv2.rectangle(img, tuple(x[0:2]), tuple(x[2:]), (0, 0, 255), 2)
    for x in annotation:
        cv2.rectangle(img, tuple(x[0:2]), tuple(x[2:]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, img_file.split("/")[-1]), img)


if __name__ == "__main__":
    data_dir = "/home/chrizandr/sports/detection_exp/annotated/"
    annotation_dir = "/home/chrizandr/sports/detection_exp/annotations/"

    output_dir = "/home/chrizandr/detection/res101_pascal_out/"
    size = (720, 1280)

    # range = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    range = [0.5]
    # range = [0.25]

    blue = []
    red = []
    white = []
    green = []
    yellow = []

    view_keys = ["top_in", "top_out", "ground_in", "ground_out"]
    str_keys = ["fb", "fa", "ps", "bm"]

    # Total AP
    ap = map_voc_range(output_dir, data_dir, annotation_dir, range, size=size)
    print("Total:", ap)

    # view wise
    for k in view_keys:
        ap = map_voc_range(output_dir, data_dir, annotation_dir, range, filter_str1=k, size=size)
        print(k + ":", ap)

    # Color wise
    for k in str_keys:
        ap = map_voc_range(output_dir, data_dir, annotation_dir, range, filter_str1=k, size=size)
        if k == "fb":
            blue.append(ap)
            red.append(ap)
        if k == "ps":
            red.append(ap)
            white.append(ap)
        if k == "fa":
            blue.append(ap)
            white.append(ap)
        if k == "bm":
            green.append(ap)
            yellow.append(ap)

    print("blue: ", sum(blue)/len(blue))
    print("red: ", sum(red)/len(red))
    print("white: ", sum(white)/len(white))
    print("green: ", sum(green)/len(green))
    print("yellow: ", sum(yellow)/len(yellow))

    # Color and view wise
    for k1 in view_keys:
        for k in str_keys:
            ap = map_voc_range(output_dir, data_dir, annotation_dir, range, filter_str1=k, size=size, filter_str2=k1)
            if k == "fb":
                blue.append(ap)
                red.append(ap)
            if k == "ps":
                red.append(ap)
                white.append(ap)
            if k == "fa":
                blue.append(ap)
                white.append(ap)
            if k == "bm":
                green.append(ap)
                yellow.append(ap)
        print("------", k1, "------")
        print("blue: ", sum(blue)/len(blue))
        print("red: ", sum(red)/len(red))
        print("white: ", sum(white)/len(white))
        print("green: ", sum(green)/len(green))
        print("yellow: ", sum(yellow)/len(yellow))
