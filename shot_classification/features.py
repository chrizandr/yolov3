# import tqdm
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


def compute_features(img_dir, save_individual=False, save_folder=""):
    """Compute features for all images."""
    files = os.listdir(img_dir)
    if not save_individual:
        descriptors = {}
        for i, f in enumerate(files):
            print("Processing file", i)
            if f.endswith(".png") or f.endswith(".jpg"):
                img = cv.imread(os.path.join(img_dir, f))
                kp, des = get_SIFT(img)
                indices = list(range(len(kp)))
                indices.sort(key=lambda x: kp[x].response, reverse=True)
                des = des[indices]
                descriptors[f] = des
        return descriptors
    else:
        for i, f in enumerate(files):
            print("Processing file", i)
            if f.endswith(".png") or f.endswith(".jpg"):
                img = cv.imread(os.path.join(img_dir, f))
                kp, des = get_SIFT(img)
                indices = list(range(len(kp)))
                indices.sort(key=lambda x: kp[x].response, reverse=True)
                des = des[indices]
                file = open(os.path.join(save_folder, f.replace(".png", ".pkl").replace(".jpg", ".pkl")), "wb")
                pickle.dump(des, file)
                file.close()
        return None


if __name__ == "__main__":
    images = "/ssd_scratch/cvit/chrizandr/images/"
    save_folder = "/ssd_scratch/cvit/chrizandr/features2"
    descriptors = compute_features(images, save_individual=True, save_folder=save_folder)
    # file = open("features.pkl", "wb")
    # pickle.dump(descriptors, file)
    pdb.set_trace()
