import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import pickle


def create_fig(num_curves: int):
    fig = plt.figure()
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(num_curves, 1),
        axes_pad=0.05,
        share_all=True,
        # label_mode="1",
        aspect=False,
    )
    # grid[0].get_yaxis().set_ticks([])
    # grid[0].get_xaxis().set_ticks([])
    return fig, grid


def main():

    in_files = [
        ("tmp/forecast_WaveNet.pkl", "model A"),
        ("tmp/forecast_WaveNet.pkl", "model B"),
    ]
    cds = [pickle.loads(open(p[0], "rb").read()) for p in in_files]
    num_curves = cds[0]["curves"]["obs"]["values"].shape[0]

    fig, grid = create_fig(num_curves)

    # Plot obs
    c_obs = cds[0]["curves"]["obs"]
    t_obs = np.arange(c_obs["t_range"][0], c_obs["t_range"][1], 1)
    for y, ax in zip(c_obs["values"], grid):
        h = ax.step(t_obs, y, linewidth=0.5, linestyle="--", alpha=0.8, c="k")
        ax.set_ylim(0, cds[0]["qlevels"])
    handles = [h[0]]
    labels = ["Input"]

    # Plot generated data
    for fdata, c in zip(in_files, cds):
        c_gen = c["curves"]["gen"]
        t_gen = np.arange(c_gen["t_range"][0], c_gen["t_range"][1], 1)
        for y, ax in zip(c_gen["values"], grid):
            h = ax.step(t_gen, y)
        handles += [h[0]]
        labels += [fdata[1]]

    fig.legend(handles, labels, loc="upper center", ncol=(len(in_files) + 1))
    fig.tight_layout()
    fig.savefig(f"tmp/forecast_compare.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()