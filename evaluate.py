"""Evaluate results and find the reasons for errors."""
import numpy as np
import os
import pdb
from soccer_annotations import get_scaled_annotations_ball, get_scaled_annotations_person


def read_output_file(filename):
    """Read the output files for predictions."""
    annotations = []
    with open(filename, "r") as f:
        for line in f:
            d = line.split()
            d = [int(float(x)) for x in d[0:4]]
            annotations.append(d)
    return np.array(annotations)


def evaluate_person(output_dir, data_dir, annotation_file, overlap=0.75):
    """Evaluate output for the player data."""
    files = os.listdir(data_dir)
    annot_files = os.listdir(output_dir)
    output_imgs = [x.strip(".txt") for x in annot_files]
    annotations = get_scaled_annotations_person(annotation_file)
    correct = 0
    total = 0
    for f in files:
        annotation = annotations[f]
        total = total + annotation.shape[0]
        if f in output_imgs:
            output = read_output_file(os.path.join(output_dir, f + ".txt"))
            count = match_annotations(output, annotation, overlap)
            correct += count
    return float(correct)/total * 100


def evaluate_ball(output_dir, data_dir, annotation_dir, overlap=0.75):
    """Evaluate output for the ball data."""
    files = os.listdir(data_dir)
    annot_files = os.listdir(output_dir)
    output_imgs = [x.strip(".txt") for x in annot_files]
    annotations = get_scaled_annotations_ball(annotation_dir)
    pdb.set_trace()
    correct = 0
    total = 0
    for f in files:
        annotation = annotations[f]
        total = total + annotation.shape[0]
        if f in output_imgs:
            output = read_output_file(os.path.join(output_dir, f + ".txt"))
            count = match_annotations(output, annotation, overlap)
            correct += count
    return float(correct)/total * 100


def match_annotations(output, annotation, overlap):
    """Match the annotations and output in the image and find accuracy."""
    count = 0
    for o in output:
        # pdb.set_trace()
        dist = np.sum((annotation - o) ** 2, axis=1)
        closest_box = annotation[np.argmin(dist), :]
        iou = bb_intersection_over_union(closest_box, o)
        if iou >= overlap:
            count += 1
    return count


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


if __name__ == "__main__":
    player_data = "/home/chrizandr/sports/SoccerPlayerDetection_bmvc17_v1/DataSet_002/"
    soccer_ball_data = "/home/chrizandr/sports/soccer_ball_data/images/"
    output_player = "output_person/"
    output_soccer = "output_ball/"
    annotation_file = "/home/chrizandr/sports/SoccerPlayerDetection_bmvc17_v1/annotation_2.mat/"
    annotation_dir = "/home/chrizandr/sports/soccer_ball_data/annotations/"

    evaluate_ball(output_soccer, soccer_ball_data, annotation_dir)

    # olap = np.linspace(0, 1, 10)
    # acc = [evaluate_person(output_dir, player_data, annotation_file, x) for x in olap]
    #
    # import matplotlib.pyplot as plt
    # plt.plot(olap, acc)
    # plt.show()