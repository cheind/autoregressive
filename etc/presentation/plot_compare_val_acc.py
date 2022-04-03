import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import json


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
        ("etc/presentation/run-version_71-tag-val_acc_epoch.json", "1-step training"),
        ("etc/presentation/run-version_70-tag-val_acc_epoch.json", "8-step training"),
    ]
    vaccs = [np.asarray(json.loads(open(p[0], "r").read())) for p in in_files]
    print(vaccs)

    fig, ax = plt.subplots()
    for fdata, vacc in zip(in_files, vaccs):
        ax.plot(vacc[:, 1], vacc[:, 2], label=fdata[1])
    ax.set_ylabel("mean val. acc (8-step RO)")
    ax.set_xlabel("training step")
    fig.legend(loc="upper center", ncol=(len(in_files)))
    fig.savefig(f"tmp/compare_val_acc.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
