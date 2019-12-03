import numpy as np
import os
import matplotlib.pyplot as plt
import pdb


def read_output_file(filepath, save_prefix="graph", train_line=True):
    f = open(filepath, "r")
    res = []
    for line in f:
        l_ = line.split()
        if train_line:
            if l_[0] == "Val":
                res.append([float(x) for x in l_[3::]])
        else:
            res.append([float(x) for x in l_[2::]])

    res = np.array(res)
    train_res = res[::, 0:5]
    val_res = res[::, 7::]
    plot_train(train_res, save_prefix)
    plot_train_val(train_res, val_res, save_prefix)


def plot_train(train_res, save_prefix):
    loss_xy, loss_wh, loss_conf, loss_cls, loss_total = train_res[::, 0:5].T
    epochs = np.arange(train_res.shape[0])
    plt.plot(epochs, loss_xy, label="loss_xy")
    plt.plot(epochs, loss_wh, label="loss_wh")
    plt.plot(epochs, loss_conf, label="loss_conf")
    plt.plot(epochs, loss_cls, label="loss_cls")
    plt.plot(epochs, loss_total, label="loss_total")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Individual training losses")
    plt.legend()
    plt.savefig(save_prefix + "_train_loss.png")
    plt.close()


def plot_train_val(train_res, val_res, save_prefix):
    t_loss_total = train_res[::, 4]
    v_loss_total = val_res[::, -1]
    epochs = np.arange(train_res.shape[0])
    plt.plot(epochs, t_loss_total, label="Training loss")
    plt.plot(epochs, v_loss_total, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training + Validation loss")
    plt.savefig(save_prefix + "_val_loss.png")
    plt.close()

    v_P, v_R, v_map, v_f1 = val_res[::, 0:4].T
    plt.plot(epochs, v_P, label="Precision")
    plt.plot(epochs, v_R, label="Recall")
    plt.plot(epochs, v_map, label="mAP")
    plt.plot(epochs, v_f1, label="F1")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Validation Accuracy")
    plt.savefig(save_prefix + "_val_acc.png")
    plt.close()


if __name__ == "__main__":
    output_file = "results2.txt"
    save_prefix = "yolo_synth_mod"
    train_line = True

    read_output_file(output_file, save_prefix, train_line)
