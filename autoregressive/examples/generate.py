import itertools
import time

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from .. import dataset, variants, wave

MODEL_CLASSES = {"RegressionWaveNet": variants.RegressionWaveNet}


@torch.no_grad()
def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "-variant",
        choices=list(MODEL_CLASSES.keys()),
        default=list(MODEL_CLASSES.keys())[0],
    )
    parser.add_argument("-num-traj", type=int, default=1)
    parser.add_argument("-num-steps", type=int, default=256)
    parser.add_argument("-num-curves", type=int, default=4)
    parser.add_argument(
        "--fast-wavenet", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("ckpt", type=Path)
    args = parser.parse_args()
    assert args.ckpt.is_file()

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = MODEL_CLASSES[args.variant].load_from_checkpoint(args.ckpt)
    model = model.eval().to(dev)
    sampler = model.create_sampler()

    _, dataset_val = dataset.create_default_datasets()

    fig = plt.figure()
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(args.num_curves, 1),
        axes_pad=0.05,
        share_all=True,
        label_mode="1",
    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    # Get samples from dataset. These are n-dicts
    curves = list(itertools.islice(dataset_val, args.num_curves))

    # Prepare observations. We batch all observations and then repeat
    # these observations for the number of trajectories
    obs = torch.stack([c["x"] for c in curves], 0).unsqueeze(1)  # (B,1,T)
    obs = obs.repeat(args.num_traj, 1, 1).to(dev)

    # Generate. Note, we leave the last obs out so the first item predicted
    # overlaps the last observation and we hence get a smooth plot
    if args.fast_wavenet:
        g = wave.generate_fast(
            model=model,
            initial_obs=obs[..., :-1],
            sampler=sampler,
        )
    else:
        g = wave.generate(
            model=model,
            initial_obs=obs[..., :-1],
            sampler=sampler,
        )

    t0 = time.time()
    trajs = torch.cat(list(itertools.islice(g, args.num_steps)), -1).squeeze(1).cpu()
    print(f"Generation took {(time.time()-t0):.3f} secs")

    # Plot
    for idx, (ax, s) in enumerate(zip(grid, curves)):
        xo, t = s["xo"], s["t"]
        dt = t[-1] - t[-2]
        tn = torch.arange(0.0, dt * args.num_steps, dt) + t[-1]

        ax.plot(t, xo, c="k", linewidth=0.5, label="input")
        for tidx, xn in enumerate(trajs[idx :: args.num_curves]):
            # Note, above we step with num_curves to get all trajectories
            # for this axis. Related to repeat statement above.
            ax.plot(tn, xn, label="generated" if tidx == 0 else "")
        ax.set_ylim(-3, 3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    fig.suptitle(f"Sample generation by {args.variant}")
    fig.savefig(f"tmp/generate_{args.variant}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
