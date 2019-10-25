
import numpy as np
import os
import pdb
import pickle

from sklearn.cluster import KMeans


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


def gen_bow_features(model_file, feature_folder, dist_file, num_words):
    """Get bow distribution for each image."""
    print("Reading model")
    model = pickle.load(open(model_file, "rb"))
    print("Reading data")
    files = os.listdir(feature_folder)
    bins = list(range(num_words+1))

    distribution = {}

    for f in files:
        if f.endswith(".pkl"):
            print("Processing file", f)
            des = pickle.load(open(os.path.join(feature_folder, f), "rb"), encoding="latin1")
            labels = model.predict(des)
            dist, bins_ = np.histogram(labels, bins)
            distribution[f.replace(".pkl", ".png")] = dist

    file = open(dist_file, "wb")
    pickle.dump(distribution, file)
    return distribution


if __name__ == "__main__":
    data_folder = "/ssd_scratch/cvit/chrizandr/images/"
    feature_folder = "/ssd_scratch/cvit/chrizandr/features2/"
    model_file = "bof_model.pkl"
    feature_file = "features.pkl"
    dist_file = "distribution.pkl"
    num_words = 150

    # train_bof_model(feature_file, model_file, num_words=num_words)
    distribution = gen_bow_features(model_file, feature_folder, dist_file, num_words)
