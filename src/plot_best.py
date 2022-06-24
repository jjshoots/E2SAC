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


def get_log_from_uri(uri, keys, api=None):
    assert isinstance(keys, list), "keys must be a list."
    api = wandb.Api(timeout=30) if api is None else api
    run = api.run(uri)
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        array = array.astype(np.float64)
        array = np.nan_to_num(array, nan=-100.0, posinf=-100.0, neginf=-100.0)
        data[key] = array

    return data


def get_log_from_run(run, keys):
    assert isinstance(keys, list), "keys must be a list."
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
    num_steps = 200000
    num_intervals = 40

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # collect runs from sweeps
    api = wandb.Api(timeout=30)
    runs = {}
    runs["SAC"] = api.sweep("jjshoots/carracing_sweep2/kibpvwiz").runs
    runs["CCGE"] = api.sweep("jjshoots/carracing_sweep2/9fecuh42").runs

    # list of algorithms we have
    algorithms = [key for key in runs]

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    scores = {}
    for algorithm in runs:
        score = []
        for run in runs[algorithm]:
            log = get_log_from_run(run, ["num_transitions", "eval_perf"])
            if algorithm == "CCGE2":
                score.append(
                    np.interp(x_axis, log["num_transitions"], log["eval_perf"])
                )
            else:
                score.append(
                    np.interp(x_axis, log["num_transitions"], log["eval_perf"] * 1.1875)
                )

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
