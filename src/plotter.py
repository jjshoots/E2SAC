# rliable google
import matplotlib.patches as patches

# plotting styles
import matplotlib.pyplot as plt

# normal imports
import numpy as np
import seaborn as sns
from matplotlib import rc, rcParams
from rliable import library as rly
from rliable import metrics, plot_utils

# to import stuff from wandb
import wandb

sns.set_style("white")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

# See warnings only once
import warnings

warnings.filterwarnings("default")

# The answer to life, universe and everything
RAND_STATE = np.random.RandomState(42)

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

rc("text", usetex=False)


def get_wandb_log(run_uri, keys):
    assert isinstance(keys, list), "keys must be a list."
    api = wandb.Api(timeout=30)
    run = api.run(run_uri)
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        array = array.astype(np.float64)
        array = np.nan_to_num(array, nan=-100.0, posinf=-100.0, neginf=-100.0)
        data[key] = array

    return data


if __name__ == "__main__":
    # parameters
    num_steps = 150000
    num_intervals = 40

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris
    runs = {}
    runs["SAC"] = [
        "jjshoots/carracing_sweep2/w3ogo5i3",
        "jjshoots/carracing_sweep2/axsl1n6b",
        "jjshoots/carracing_sweep2/0fna6aux",
        "jjshoots/carracing_sweep2/i6p1v3j1",
        "jjshoots/carracing_sweep2/8yttgptr",
        "jjshoots/carracing_sweep2/r8rthkfe",
        "jjshoots/carracing_sweep2/mgoadwrf",
        "jjshoots/carracing_sweep2/herqnh49",
        "jjshoots/carracing_sweep2/1cv736lf",
        "jjshoots/carracing_sweep2/hneliap7",
        "jjshoots/carracing_sweep2/i2cddx3i",
        "jjshoots/carracing_sweep2/js786awl",
        "jjshoots/carracing_sweep2/8hvktctd",
        "jjshoots/carracing_sweep2/icoqszdi",
        "jjshoots/carracing_sweep2/lcuxfa72",
        "jjshoots/carracing_sweep2/uzth8si1",
        "jjshoots/carracing_sweep2/55u59b54",
        "jjshoots/carracing_sweep2/yaom6v8d",
        "jjshoots/carracing_sweep2/2qk1w54c",
        "jjshoots/carracing_sweep2/3js1el59",
        "jjshoots/carracing_sweep2/kezq2tqx",
        "jjshoots/carracing_sweep2/mn67a7i5",
        "jjshoots/carracing_sweep2/e6aib2wo",
        "jjshoots/carracing_sweep2/9uo8kmp6",
        "jjshoots/carracing_sweep2/djmpwti5",
        "jjshoots/carracing_sweep2/0ud8k3lo",
        "jjshoots/carracing_sweep2/068nxo3c",
        "jjshoots/carracing_sweep2/qnjgsadb",
        "jjshoots/carracing_sweep2/oq98lv5a",
        "jjshoots/carracing_sweep2/dcec16qz",
        "jjshoots/carracing_sweep2/ug59gfg5",
        "jjshoots/carracing_sweep2/tb0yq03j",
        "jjshoots/carracing_sweep2/2vfzdqkx",
        "jjshoots/carracing_sweep2/f9im5dwu",
        "jjshoots/carracing_sweep2/q73d7uba",
        "jjshoots/carracing_sweep2/w2hruqzn",
        "jjshoots/carracing_sweep2/cbwsvbhw",
        "jjshoots/carracing_sweep2/yji6t5e3",
        "jjshoots/carracing_sweep2/u5v8h19i",
        "jjshoots/carracing_sweep2/e1s6cbe1",
        "jjshoots/carracing_sweep2/n2lxg46q",
        "jjshoots/carracing_sweep2/wgfo4rsi",
        "jjshoots/carracing_sweep2/pfku7gc6",
        "jjshoots/carracing_sweep2/f0vzlde8",
        "jjshoots/carracing_sweep2/20irw8be",
        "jjshoots/carracing_sweep2/ubwy0wqw",
        "jjshoots/carracing_sweep2/qmk8p0ra",
        "jjshoots/carracing_sweep2/768bxcu4",
        "jjshoots/carracing_sweep2/y05z0f4o",
        "jjshoots/carracing_sweep2/88s87bdy",
        "jjshoots/carracing_sweep2/cn8vzqsl",
        "jjshoots/carracing_sweep2/3wbro7j7",
        "jjshoots/carracing_sweep2/yt4eg7g8",
        "jjshoots/carracing_sweep2/j0716sdv",
        "jjshoots/carracing_sweep2/9qc4z4g5",
    ]
    runs["CCGE"] = [
        "jjshoots/carracing_sweep2/0jf87ral",
        "jjshoots/carracing_sweep2/6ga28xk3",
        "jjshoots/carracing_sweep2/530ugv6q",
        "jjshoots/carracing_sweep2/5b7o4vl0",
        "jjshoots/carracing_sweep2/85ms41vx",
        "jjshoots/carracing_sweep2/i8f4ayle",
        "jjshoots/carracing_sweep2/ggenlfrc",
        "jjshoots/carracing_sweep2/8jgqqyb2",
        "jjshoots/carracing_sweep2/yw1r9pnu",
        "jjshoots/carracing_sweep2/rv5r4m5r",
        "jjshoots/carracing_sweep2/6pmii6a2",
        "jjshoots/carracing_sweep2/5d4mveis",
        "jjshoots/carracing_sweep2/foyqfwf4",
        "jjshoots/carracing_sweep2/n1ghdy8b",
        "jjshoots/carracing_sweep2/e29j6d8l",
        "jjshoots/carracing_sweep2/hizy1trk",
        "jjshoots/carracing_sweep2/684ym2x6",
        "jjshoots/carracing_sweep2/frnye2f4",
        "jjshoots/carracing_sweep2/p6olvikx",
        "jjshoots/carracing_sweep2/v20bwawx",
        "jjshoots/carracing_sweep2/16oesko4",
        "jjshoots/carracing_sweep2/8cp5z91c",
        "jjshoots/carracing_sweep2/mlb10ce4",
        "jjshoots/carracing_sweep2/u2f2prl1",
        "jjshoots/carracing_sweep2/j0lfy15c",
        "jjshoots/carracing_sweep2/ng58ir75",
        "jjshoots/carracing_sweep2/wsq8rl19",
        "jjshoots/carracing_sweep2/8d4vzq45",
        "jjshoots/carracing_sweep2/lup8macc",
        "jjshoots/carracing_sweep2/domhzrw5",
        "jjshoots/carracing_sweep2/2r8asnyr",
        "jjshoots/carracing_sweep2/edhmmcwq",
        "jjshoots/carracing_sweep2/r4sdvzfk",
        "jjshoots/carracing_sweep2/9s9kmrzs",
        "jjshoots/carracing_sweep2/9msi8buu",
        "jjshoots/carracing_sweep2/xxsbmrlm",
        "jjshoots/carracing_sweep2/qd9qq8qp",
        "jjshoots/carracing_sweep2/yzelorz1",
        "jjshoots/carracing_sweep2/lxj88vas",
        "jjshoots/carracing_sweep2/h2w74jd6",
        "jjshoots/carracing_sweep2/cix9atte",
        "jjshoots/carracing_sweep2/5zrznky7",
        "jjshoots/carracing_sweep2/cc85cp4f",
        "jjshoots/carracing_sweep2/kna1tl82",
        "jjshoots/carracing_sweep2/p94skl7j",
        "jjshoots/carracing_sweep2/ntmh4rqv",
        "jjshoots/carracing_sweep2/qi51or2p",
        "jjshoots/carracing_sweep2/tjkic4o0",
        "jjshoots/carracing_sweep2/y1hav88s",
        "jjshoots/carracing_sweep2/94sznuj0",
        "jjshoots/carracing_sweep2/zujxlv0a",
        "jjshoots/carracing_sweep2/1oo8xblq",
    ]

    # list of algorithms we have
    algorithms = [key for key in runs]

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    scores = {}
    for algorithm in runs:
        score = []
        for run_uri in runs[algorithm]:
            log = get_wandb_log(run_uri, ["num_transitions", "eval_perf"])
            score.append(np.interp(x_axis, log["num_transitions"], log["eval_perf"]))

        # stack along num_runs axis
        score = np.stack(score, axis=0)
        # expand along num_games axis
        score = np.expand_dims(score, axis=1)

        # add to overall scores
        scores[algorithm] = score

    # get interquartile mean
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])]
    )
    # compute confidence intervals
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm, reps=50000)

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e4,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Timesteps (1e4)",
        ylabel="Evaluation Interquartile Mean (IQM)",
        labelsize=24,
        ticklabelsize=24,
    )

    plt.axhline(
        y=270, color=sns.color_palette("colorblind")[len(algorithms)], linestyle="-"
    )
    algorithms.append("Oracle")

    # form the legend
    color_dict = dict(zip(algorithms, sns.color_palette("colorblind")))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in algorithms
    ]
    legend = plt.legend(
        fake_patches,
        algorithms,
        loc="upper center",
        fancybox=True,
        # ncol=len(algorithms),
        ncol=2,
        fontsize=24,
        # handleheight=1.8,
        bbox_to_anchor=(0.5, 1.4),
    )

    # plt.title('Suboptimal Policy Eval = 270')
    # plt.savefig('resource/carracing.pdf')
    plt.subplots_adjust(top=0.7, left=0.2, bottom=0.1)
    plt.show()
