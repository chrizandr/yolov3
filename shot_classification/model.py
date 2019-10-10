"""Model."""
import numpy as np
import os
import pdb
import pickle
import cv2 as cv
import random

from sklearn.cluster import KMeans
from sklearn.svm import SVC


sift = cv.xfeatures2d.SIFT_create()


def get_SIFT(img):
    """Get SIFT features from image."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def train_bof_model(feature_file, model_file, num_words=150):
    """Train a bag of features model given a feature."""
    print("Reading data")
    data = pickle.load(open(feature_file, "rb"), encoding="latin1")
    print("Building vectors")
    raw_data = np.vstack([data[k][0:300] for k in data])
    print(raw_data.shape)

    print("Training model")
    model = KMeans(n_clusters=num_words, max_iter=2000, verbose=1, n_init=50)
    model.fit(raw_data)
    pickle.dump(model, open(model_file, "wb"))
    pdb.set_trace()


def get_bow_features(model_file, feature_file, dist_file, num_words):
    """Get bow distribution for each image."""
    print("Reading model")
    model = pickle.load(open(model_file, "rb"))
    print("Reading data")
    data = pickle.load(open(feature_file, "rb"), encoding="latin1")

    bins = list(range(num_words+1))
    distribution = {}
    for k in data:
        print("Processing file", k)
        labels = model.predict(data[k])
        dist, bins_ = np.histogram(labels, bins)
        distribution[k] = dist

    file = open(dist_file, "wb")
    pickle.dump(distribution, file)
    return distribution


def gen_bow_features(model_file, data_folder, dist_file, num_words):
    """Get bow distribution for each image."""
    print("Reading model")
    model = pickle.load(open(model_file, "rb"))
    print("Reading data")
    files = os.listdir(data_folder)
    bins = list(range(num_words+1))

    distribution = {}
    for f in files:
        print("Processing file", f)
        if f.endswith(".png") or f.endswith(".jpg"):
            img = cv.imread(os.path.join(data_folder, f))
            kp, des = get_SIFT(img)
            labels = model.predict(des)
        dist, bins_ = np.histogram(labels, bins)
        distribution[f] = dist

    file = open(dist_file, "wb")
    pickle.dump(distribution, file)
    return distribution


def get_class(name):
    """Get class label."""
    names = ["top_in", "top_out", "ground_in", "ground_out"]
    label = [n in name for n in names].index(True)
    return label


def train_model(train_files, test_files, dist_file, model_save="", density=False):
    """Train SVM for shot classification."""
    print("Reading distributions")
    data = pickle.load(open(dist_file, "rb"))
    if density:
        X_train = [data[x]/data[x].sum() for x in train_files]
        X_test = [data[x]/data[x].sum() for x in test_files]

    else:
        X_train = [data[x] for x in train_files]
        X_test = [data[x] for x in test_files]

    Y_train = [get_class(x) for x in train_files]
    Y_test = [get_class(x) for x in test_files]

    print("Training SVM")
    model = SVC(kernel="linear")
    model.fit(X_train, Y_train)

    print("Testing SVM")
    score = model.score(X_test, Y_test)
    print("Accuracy", score)

    if len(model_save) != 0:
        pickle.dump(model, open(model_save, "wb"))

    return model


def train_test_split(data_folder, split_ratio=0.7):
    """Train test split."""
    files = os.listdir(data_folder)
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    train = files[0:split_index]
    test = files[split_index::]

    return train, test


if __name__ == "__main__":
    data_folder = "/home/chrizandr/sports/detection_exp/annotated/"
    model_save = "model.pkl"
    model_file = "bof.pkl"
    feature_file = "features.pkl"
    dist_file = "distribution.pkl"
    num_words = 150

    # train_bof_model(feature_file, model_file, num_words=num_words)
    distribution = gen_bow_features(model_file, data_folder, dist_file, num_words)
    # train, test = train_test_split(data_folder)
    # model = train_model(train, test, dist_file, model_save, density=True)
    pdb.set_trace()
