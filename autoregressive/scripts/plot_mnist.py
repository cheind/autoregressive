import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from ..datasets.mnist_dataset import peano_inv_map


def main():
    data = pickle.load(open("tmp/forecast_WaveNet.pkl", "rb"))
    orig_img = torch.tensor(data["curves"]["obs"]["values"])
    gen_img = torch.tensor(data["curves"]["gen"]["values"])
    pred_img = orig_img.clone()
    tstart = data["curves"]["gen"]["t_range"][0]
    pred_img[:, tstart:] = gen_img

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(peano_inv_map(orig_img[0]))
    axs[1].imshow(peano_inv_map(pred_img[0]))
    plt.show()


if __name__ == "__main__":
    main()