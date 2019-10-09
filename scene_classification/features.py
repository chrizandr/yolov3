import os
import numpy as np
import pdb
import pickle
import cv2 as cv

sift = cv.xfeatures2d.SIFT_create()


def get_SIFT(img):
    """Get SIFT features from image."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def compute_features(img_dir):
    """Compute features for all images."""
    files = os.listdir(img_dir)
    descriptors = []
    for i, f in enumerate(files):
        print("Processing file", i)
        if f.endswith(".png") or f.endswith(".jpg"):
            img = cv.imread(os.path.join(img_dir, f))
            kp, des = get_SIFT(img)
            indices = list(range(len(kp)))
            indices.sort(key=lambda x: kp[x].response, reverse=True)
            des = des[indices]
            file = open(os.path.join("features", f.replace(".png", ".pkl")), "wb")
            pickle.dump(des[0:300], file)


if __name__ == "__main__":
    images = "/home/chris/sports/detection_exp/annotated/"
    descriptors = compute_features(images)
    pdb.set_trace()
