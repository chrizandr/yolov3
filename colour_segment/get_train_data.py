import os
import cv2
import pdb


def get_detections(data_folder, result_folder, match_string=""):
    images = os.listdir(data_folder)
    images = [x for x in images if x.endswith(".png") and match_string in x]
    detections = []
    test_images = []

    for image in images:
        result = os.path.join(result_folder, image.replace(".png", ".txt"))
        img = cv2.imread(os.path.join(data_folder, image))
        if os.path.exists(result):
            players, conf = 1   # read file here
        else:
            players, conf = [], []
        for i, p in enumerate(players):
            if conf[i] > 0.9:
                vals = image[p]  # Get pixels for detection
                # Remove detections from the image, so that no detection in test.
                images[p] = 0
        detection.append(p)
        pdb.set_trace()


if __name__ == "__main__":
    data_folder = ""
    result_folder = ""
    match_string = ""

    train_bboxes = get_detections(data_folder, result_folder)
