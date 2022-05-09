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
    num_steps = 1e6
    num_intervals = 200

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris
    runs = []
    runs.append("jjshoots/ESDDQN/211mucgg")
    runs.append("jjshoots/ESDDQN/2ximq079")
    runs.append("jjshoots/ESDDQN/3uthjuly")
    runs.append("jjshoots/ESDDQN/c2dcts1w")

    uncertainties = []
    eval_scores = []
    for run_uri in runs:
        log = get_wandb_log(run_uri, ["num_transitions", "eval_perf"])
        eval_scores.append(
            np.interp(x_axis, log["num_transitions"], log["eval_perf"])
        )

        log = get_wandb_log(run_uri, ["num_transitions", "runtime_uncertainty"])
        uncertainties.append(
            np.interp(x_axis, log["num_transitions"], log["runtime_uncertainty"])
        )

    # stack along num_runs axis
    eval_temp = np.stack(eval_scores, axis=0)
    uncer_temp = np.stack(uncertainties, axis=0)
    # expand along num_games axis
    eval_temp = np.expand_dims(eval_scores, axis=1)
    uncer_temp = np.expand_dims(uncertainties, axis=1)

    eval_scores = {}
    eval_scores['cartpole'] = eval_temp
    uncertainties = {}
    uncertainties['cartpole'] = uncer_temp

    # get interquartile mean
    iqm = lambda eval_scores: np.array(
        [metrics.aggregate_iqm(eval_scores[..., frame]) for frame in range(eval_scores.shape[-1])]
    )
    eval_iqm, eval_cis = rly.get_interval_estimates(eval_scores, iqm, reps=50000)


    # get interquartile mean
    iqm = lambda uncertainties: np.array(
        [metrics.aggregate_iqm(uncertainties[..., frame]) for frame in range(uncertainties.shape[-1])]
    )
    uncer_iqm, uncer_cis = rly.get_interval_estimates(uncertainties, iqm, reps=50000)

    # instantiate colorwheel
    palette = sns.color_palette("colorblind")
    legend_labels = {}
    i = 2

    # twin plots
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Timestep (1e6)")

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e6,
        eval_iqm,
        eval_cis,
        algorithms=None,
        xlabel=r"Timesteps (1e6)",
        ylabel="Evaluation Interquartile Mean (IQM)",
        labelsize="large",
        ticklabelsize="large",
        ax=ax1,
        custom_color=sns.color_palette("colorblind")[0:]
    )
    ax1.tick_params(axis="y", labelcolor=palette[0])
    ax1.set_ylabel("Evaluation Interquartile Mean (IQM)", color=palette[0], fontsize=20)

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e6,
        uncer_iqm,
        uncer_cis,
        algorithms=None,
        xlabel=r"Timesteps (1e6)",
        ylabel="Episodic Mean Epistemic Uncertainty",
        labelsize="large",
        ticklabelsize="large",
        ax=ax2,
        custom_color=sns.color_palette("colorblind")[1:]
    )
    ax2.tick_params(axis="y", labelcolor=palette[1])
    ax2.set_ylabel("Episodic Mean Epistemic Uncertainty", color=palette[1], fontsize=20)
    ax2.set_ylim(top=2.0)

    # legend
    # fake_patches = [
    #     patches.Patch(color=legend_labels[label], alpha=0.75) for label in legend_labels
    # ]
    # legend = plt.legend(
    #     fake_patches,
    #     legend_labels,
    #     loc="lower right",
    #     fancybox=True,
    #     # ncol=len(legend_labels),
    #     fontsize="xx-large",
    #     # bbox_to_anchor=(0.5, 1.1),
    # )

    # plt.title("Runtime Uncertainty LunarLander", fontsize="xx-large")
    plt.show()
