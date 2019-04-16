import numpy as np
import os

from soccer_annotations import get_scaled_annotations_ball, get_scaled_annotations_person


def read_output_file(filename):
    annotations = []
    with open(filename, "r") as f:
        for line in f:
            d = line.split()
            d = [int(float(x)) for x in d[0:4]]


def evaluate_person(output_dir, data_dir):
    files = os.listdir(data_dir)
    annot_files = os.listdir(output_dir)


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


if __name__ == "__main__":
