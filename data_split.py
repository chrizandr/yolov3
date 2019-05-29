"""Randomly select images with equal distribution of categories and teams."""
import os
import pdb
import random
from shutil import copyfile


if __name__ == "__main__":
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
