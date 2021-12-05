import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def main():
    data = pickle.load(open("tmp/forecast_WaveNet.pkl", "rb"))
    orig_img = torch.tensor(data["curves"]["obs"]["values"])
    gen_img = torch.tensor(data["curves"]["gen"]["values"])
    pred_img = orig_img.clone()
    tstart = data["curves"]["gen"]["t_range"][0]
    pred_img[:, tstart:] = gen_img

    N = gen_img.shape[0]
    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, nrows_ncols=(2, N), axes_pad=0.05, share_all=True, aspect=True
    )
    for (idx, (o, p)) in enumerate(zip(orig_img, pred_img)):
        grid[idx].imshow(o.view(28, 28))
        grid[N + idx].imshow(p.view(28, 28))
    plt.show()


if __name__ == "__main__":
    main()