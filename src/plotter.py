# rliable google
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# to import stuff from wandb
import wandb

# normal imports
import numpy as np
import matplotlib.patches as patches
from matplotlib import rcParams
from matplotlib import rc

# plotting styles
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

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
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        data[key] = array

    return data


if __name__ == "__main__":
    # parameters
    num_steps = 100000
    num_intervals = 21

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris
    runs = {}
    runs["SAC_CARRACING"] = [
        "jjshoots/e2SAC_carracing/1s67y2pw",
        "jjshoots/e2SAC_carracing/1tsdhfyb",
        "jjshoots/e2SAC_carracing/1zclqhsv",
        "jjshoots/e2SAC_carracing/crr57ik4",
        "jjshoots/e2SAC_carracing/j2hc8xu6",
        "jjshoots/e2SAC_carracing/18emr0dv",
        "jjshoots/e2SAC_carracing/1wuqkevm",
        "jjshoots/e2SAC_carracing/29ttdvib",
        "jjshoots/e2SAC_carracing/2fmk9hyk",
        "jjshoots/e2SAC_carracing/2hh3ouwy",
        "jjshoots/e2SAC_carracing/32j5589r",
        "jjshoots/e2SAC_carracing/33zqwfrr",
        "jjshoots/e2SAC_carracing/36qn6nxt",
        "jjshoots/e2SAC_carracing/3layv5zm",
    ]
    runs["E2SAC_CARRACING_20"] = [
        "jjshoots/e2SAC_carracing/15ox79xj",
        "jjshoots/e2SAC_carracing/1zcw8otp",
        "jjshoots/e2SAC_carracing/2o3ln6sc",
        "jjshoots/e2SAC_carracing/10ka1fdp",
        "jjshoots/e2SAC_carracing/3ubd5084",
        "jjshoots/e2SAC_carracing/y0ku3c13",
        "jjshoots/e2SAC_carracing/39euchd8",
        "jjshoots/e2SAC_carracing/3pdurrin",
        "jjshoots/e2SAC_carracing/396chdhr",
        "jjshoots/e2SAC_carracing/gh52f025",
        "jjshoots/e2SAC_carracing/3vhzqg7g",
        "jjshoots/e2SAC_carracing/384aiwc7",
        "jjshoots/e2SAC_carracing/3gwqv4sy",
        "jjshoots/e2SAC_carracing/1yfoik4h",
        "jjshoots/e2SAC_carracing/32rhqm2g",
        "jjshoots/e2SAC_carracing/upu5fgds",
    ]
    runs["E2SAC_CARRACING_8"] = [
        "jjshoots/e2SAC_carracing/2smcyzzs",
        "jjshoots/e2SAC_carracing/1mn3fmw9",
        "jjshoots/e2SAC_carracing/2sn5b8b5",
        "jjshoots/e2SAC_carracing/16t2whq1",
        "jjshoots/e2SAC_carracing/26yr8o25",
        "jjshoots/e2SAC_carracing/3puj87ig",
        "jjshoots/e2SAC_carracing/1cvkql33",
        "jjshoots/e2SAC_carracing/270g98tg",
        "jjshoots/e2SAC_carracing/fr1k6g8a",
    ]
    runs["E2SAC_CARRACING_2"] = [
        "jjshoots/e2SAC_carracing/1u6qf5r4",
        "jjshoots/e2SAC_carracing/1ss3yrfd",
        "jjshoots/e2SAC_carracing/393700q4",
        "jjshoots/e2SAC_carracing/1ibxhnm2",
        "jjshoots/e2SAC_carracing/1kmxayj6",
        "jjshoots/e2SAC_carracing/2jm10wyp",
        "jjshoots/e2SAC_carracing/1xxp2n00",
        "jjshoots/e2SAC_carracing/1y2dpmxy",
        "jjshoots/e2SAC_carracing/2m25e2h6",
        "jjshoots/e2SAC_carracing/33fepo9b",
        "jjshoots/e2SAC_carracing/3a4mpnlp",
        "jjshoots/e2SAC_carracing/3mb5x3wx",
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
        x_axis,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Number of episodes",
        ylabel="Interquartile Mean (IQM)",
    )

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
        ncol=len(algorithms),
        fontsize="x-large",
        bbox_to_anchor=(0.5, 1.1),
    )

    plt.title('Suboptimal Policy Eval = 270')
    plt.savefig('resource/carracing.pdf')
    plt.show()
