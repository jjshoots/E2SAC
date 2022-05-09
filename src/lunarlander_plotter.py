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
        array = np.nan_to_num(array, nan=-200.0, posinf=-200.0, neginf=-200.0)
        data[key] = array

    return data


if __name__ == "__main__":
    # parameters
    num_steps = 400000
    num_intervals = 200

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris
    run = "jjshoots/ESDDQN/2mlelzjg"
    span_region = [200000, 275000]
    # run = "jjshoots/ESDDQN/1qtbr3np"
    # span_region = [150000, 275000]

    log = get_wandb_log(run, ["num_transitions", "runtime_uncertainty", "eval_perf"])
    eval_score = np.interp(x_axis, log["num_transitions"], log["eval_perf"])
    uncertainty = np.interp(x_axis, log["num_transitions"], log["runtime_uncertainty"])

    # instantiate colorwheel
    palette = sns.color_palette("colorblind")
    legend_labels = {}
    i = 2

    # twin plots
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Timesteps (1e5)", fontsize="xx-large")

    # plot the eval score
    legend_labels["Evaluation Score"] = palette[i + 0]
    ax1.set_ylabel("Evaluation Score", color=palette[i + 0], fontsize=20)
    ax1.plot(x_axis / 1e5, eval_score, color=palette[i + 0])
    ax1.tick_params(axis="y", labelcolor=palette[i + 0])

    # plot the uncertainty
    legend_labels["Episodic Mean Epistemic Uncertainty"] = palette[i + 1]
    ax2.set_ylabel(
        "Episodic Mean Epistemic Uncertainty", color=palette[i + 1], fontsize=20
    )
    ax2.plot(x_axis / 1e5, uncertainty, color=palette[i + 1])
    ax2.tick_params(axis="y", labelcolor=palette[i + 1])

    # fill in the area where the agent discovers things
    legend_labels["Learns Landing"] = palette[i + 2]
    ax1.axvspan(
        span_region[0] / 1e5, span_region[1] / 1e5, alpha=0.5, color=palette[i + 2]
    )

    # legend
    fake_patches = [
        patches.Patch(color=legend_labels[label], alpha=0.75) for label in legend_labels
    ]
    legend = plt.legend(
        fake_patches,
        legend_labels,
        loc="lower right",
        fancybox=True,
        # ncol=len(legend_labels),
        fontsize="xx-large",
        # bbox_to_anchor=(0.5, 1.1),
    )

    # plt.title("Runtime Uncertainty LunarLander", fontsize="xx-large")
    plt.show()
    # plt.savefig(f"resource/lunarlander_uncertainty2.pdf")
