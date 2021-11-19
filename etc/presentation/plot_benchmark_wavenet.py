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
    data = pickle.loads(
        open("etc/presentation/profile_generators_W64_B32_Q8_N256.pkl", "rb").read()
    )
    rs = [d["L"] for d in data]
    slow = [d["slow"] * 100 for d in data]
    fast = [d["fast"] * 100 for d in data]

    fig, ax = plt.subplots()
    ax.plot(rs, slow, label="default generation")
    ax.plot(rs, fast, label="fast generation")
    ax.set_ylabel("Runtime [sec] for 100 samples")
    ax.set_xlabel("Number of Dilated Layers")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(f"tmp/benchmark_generators.svg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()