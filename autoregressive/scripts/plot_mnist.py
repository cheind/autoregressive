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

    N = gen_img.shape[0]
    fig, axs = plt.subplots(N, 2)
    for (
        idx,
        (o, p),
    ) in enumerate(zip(orig_img, pred_img)):
        axs[idx, 0].imshow(peano_inv_map(o))
        axs[idx, 1].imshow(peano_inv_map(p))
        axs[idx, 0].set_aspect("auto")
        axs[idx, 1].set_aspect("auto")
    plt.show()


if __name__ == "__main__":
    main()