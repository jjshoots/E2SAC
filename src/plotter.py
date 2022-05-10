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
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        data[key] = array

    return data


if __name__ == "__main__":
    # parameters
    num_steps = 500000
    num_intervals = 200

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris
    runs = {}
    runs["Runtime Uncertainty"] = [
        "jjshoots/carracing_discrete/16ghiz19",
        "jjshoots/carracing_discrete/1t1i6i4z",
        "jjshoots/carracing_discrete/ifbtm01s",
        "jjshoots/carracing_discrete/2e01yqe5",
        "jjshoots/carracing_discrete/q4mciaiy",
        "jjshoots/carracing_discrete/evvbcntl",
        "jjshoots/carracing_discrete/2ad054yx",
        "jjshoots/carracing_discrete/85wv07fe",
        "jjshoots/carracing_discrete/11d3jqat",
        "jjshoots/carracing_discrete/1jla2e50",
        "jjshoots/carracing_discrete/2otv5zh6",
        "jjshoots/carracing_discrete/11d3jqat",
    ]

    # list of algorithms we have
    algorithms = [key for key in runs]

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    scores = {}
    for algorithm in runs:
        score = []
        for run_uri in runs[algorithm]:
            log = get_wandb_log(run_uri, ["num_transitions", "runtime_uncertainty"])
            score.append(np.interp(x_axis, log["num_transitions"], log["runtime_uncertainty"]))

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
        x_axis / 1e5,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Timesteps (1e5)",
        ylabel="Episodic Mean Epistemic Uncertainty",
        labelsize=24,
        ticklabelsize=24,
    )

    plt.axvline(x=250000/1e5, color=sns.color_palette("colorblind")[1], linestyle="-")
    algorithms.append("Domain Change")

    # form the legend
    color_dict = dict(zip(algorithms, sns.color_palette("colorblind")))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in algorithms
    ]
    legend = plt.legend(
        fake_patches,
        algorithms,
        loc="lower right",
        fancybox=True,
        ncol=len(algorithms),
        fontsize="xx-large",
        # bbox_to_anchor=(0.5, 1.1),
    )

    # bound limites
    plt.ylim(top=25)

    # plt.title('Runtime Uncertainty CarRacing-v1', fontsize="xx-large")
    # plt.savefig('resource/RuntimeUncertaintyCarRacing.pdf')
    plt.show()
