"""Randomly select images with equal distribution of categories and teams."""
import os
import pdb
import random
from shutil import copyfile


def select_test_set():
    main_folder = "/home/chrizandr/sports/detection_exp/"
    output_folder = "/home/chrizandr/sports/detection_exp/annotated/"
    categories = ["ground_in", "ground_out", "top_in", "top_out"]
    matches = ["bm", "fa", "fb", "ps"]

    final_files = []
    for c in categories:
        files = os.listdir(os.path.join(main_folder, c))
        random.shuffle(files)
        for m in matches:
            counter = 0
            for f in files:
                if m in f:
                    final_files.append((c, f))
                    counter += 1
                if counter == 25:
                    break

    for c, f in final_files:
        src = os.path.join(main_folder, c, f)
        dest = os.path.join(output_folder, c + "_" + f)
        print("processing", f)
        copyfile(src, dest)


def split_training_data(data_folder, prefix="", split=0):
    files = os.listdir(data_folder)
    files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
    if len(prefix) == 0:
        prefix = data_folder

    files = [os.path.join(prefix, x) for x in files]
    random.shuffle(files)
    split_idx = int(len(files) * split)
    train = files[0: split_idx]
    val = files[split_idx::]

    with open("train.txt", "w") as f:
        f.write("\n".join(train))
    with open("val.txt", "w") as f:
        f.write("\n".join(val))


if __name__ == "__main__":
    data_folder = "/home/chrizandr/sports/detection_exp/images/"
    split_training_data(data_folder)
