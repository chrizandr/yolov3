"""Model."""
import numpy as np
import os
from shutil import copyfile
import pdb
import pickle
import random

from sklearn.svm import SVC


def save_results(preds, files, data_folder, save_place, save_as_files=False):
    """Save results after prediction."""
    if save_as_files:
        assert os.path.isdir(save_place)
        for i, f in enumerate(files):
            print("Saving file", f)
            class_name = get_class(preds[i], reverse=True)
            save_file_name = os.path.join(save_place, class_name + "_" + f)
            source_file_name = os.path.join(data_folder, f)
            copyfile(source_file_name, save_file_name)
    else:
        num_files = len(preds)
        save_text = "\n".join([",".join([get_class(preds[i], reverse=True), files[i]])
                               for i in range(num_files)])
        f = open(save_place, "w")
        f.write(save_text)
        f.close()


def get_class(name, reverse=False):
    """Get class label."""
    names = ["top_in", "top_out", "ground_in", "ground_out"]
    if reverse:
        assert type(name) is not str
        label = names[name]
    else:
        label = [n in name for n in names].index(True)
    return label


def smoothen_distribution(frame, window, data, density):
    frame_id = int(frame.split(".")[0].split("_")[-1])
    frame_prefix = "_".join(frame.split(".")[0].split("_")[0:-1])
    window_frames = [frame_prefix + "_" + str(x) + ".png"
                     for x in range(frame_id-window//2, frame_id+window//2)
                     if frame_prefix + "_" + str(x) + ".png" in data]
    frames_data = [data[x] for x in window_frames]
    try:
        assert len(window_frames) != 0
    except AssertionError:
        pdb.set_trace()
    des = sum(frames_data)
    if density:
        des = des/des.sum()
    return des


def train_model(train_files, test_files, dist_file, window=40, model_save="", density=False):
    """Train SVM for shot classification."""
    print("Reading distributions")
    data = pickle.load(open(dist_file, "rb"))
    if window != 0:
        X_train = [smoothen_distribution(x, window, data, density) for x in train_files]
        X_test = [smoothen_distribution(x, window, data, density) for x in test_files]
    else:
        if density:
            X_train = [data[x]/data[x].sum() for x in train_files]
            X_test = [data[x]/data[x].sum() for x in test_files]
        else:
            X_train = [data[x] for x in train_files]
            X_test = [data[x] for x in test_files]

    Y_train = [get_class(x) for x in train_files]
    Y_test = [get_class(x) for x in test_files]

    X_train = np.array(X_train)
    X_test = np.array(X_test)

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


def evaluate(dist_file, model_file, data_folder, save_place="results.txt", window=40, density=False):
    """Do shot classification."""
    print("Load distribution")
    data = pickle.load(open(dist_file, "rb"))
    files = list(data.keys())

    if window != 0:
        X = [smoothen_distribution(x, window, data, density) for x in files]
    else:
        if density:
            X = [data[x]/data[x].sum() for x in files]
        else:
            X = [data[x] for x in files]

    X = np.array(X)
    model = pickle.load(open(model_file, "rb"))
    preds = model.predict(X)

    save_results(preds, files, data_folder, save_place, save_as_files=True)


if __name__ == "__main__":
    data_folder = "/ssd_scratch/cvit/chrizandr/images/"
    shot_model = "shot_model.pkl"
    bof_model = "bof_model.pkl"
    dist_file = "1300_distribution.pkl"
    result_save = "/home/chrizandr/sports/test/images2/"
    num_words = 150
    window = 40

    # train, test = train_test_split(data_folder)
    # model = train_model(train, test, dist_file, window=window, model_save=shot_model, density=False)
    evaluate(dist_file, shot_model, data_folder, result_save, window)
    pdb.set_trace()
