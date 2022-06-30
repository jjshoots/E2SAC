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


def get_wandb_log(run, keys, start_val):
    assert isinstance(keys, list), "keys must be a list."
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        array = array.astype(np.float64)
        array = np.nan_to_num(array, nan=start_val, posinf=start_val, neginf=start_val)
        data[key] = array

    return data


def process_sweep(sweep_uri, sweep_name, start_val=-100):
    # parameters
    num_steps = 100000
    num_intervals = 20

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # get the sweep handler
    sweep = wandb.Api(timeout=30).sweep(sweep_uri)

    # list for legend later
    legend_list = []
    legend_list.append(sweep_name)

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    uncer_list = []
    eval_list = []
    for run in sweep.runs:
        log = get_wandb_log(
            run, ["num_transitions", "eval_perf", "runtime_uncertainty"], start_val
        )
        if len(log["eval_perf"]) > 0 and len(log["runtime_uncertainty"] > 0):
            eval_list.append(
                np.interp(x_axis, log["num_transitions"], log["eval_perf"])
            )

            uncer_list.append(
                np.interp(x_axis, log["num_transitions"], log["runtime_uncertainty"])
            )

    # expand along num_games axis
    eval_scores = {}
    eval_scores[sweep_name] = np.expand_dims(eval_list, axis=1)
    uncertainties = {}
    uncertainties[sweep_name] = np.expand_dims(uncer_list, axis=1)

    # get interquartile mean
    iqm = lambda uncertainties: np.array(
        [
            metrics.aggregate_iqm(uncertainties[..., frame])
            for frame in range(uncertainties.shape[-1])
        ]
    )
    # compute confidence intervals
    iqm_scores, iqm_cis = rly.get_interval_estimates(uncertainties, iqm, reps=50000)

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e3,
        iqm_scores,
        iqm_cis,
        algorithms=None,
        xlabel=r"Timesteps (1e3)",
        ylabel="Episodic Mean F-value",
        labelsize=24,
        ticklabelsize=24,
    )

    # draw the vertical line at domain change
    legend_list.append("Domain Change")
    plt.axvline(x=50000 / 1e3, color=sns.color_palette("colorblind")[len(legend_list) - 1], linestyle="-")

    # form the legend
    color_dict = dict(zip(legend_list, sns.color_palette("colorblind")))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in legend_list
    ]
    legend = plt.legend(
        fake_patches,
        legend_list,
        loc="lower center",
        fancybox=True,
        # ncol=len(legend_list),
        fontsize=24,
        # bbox_to_anchor=(0.5, 1.1),
    )

    plt.title('CarRacing w/ Domain Change F-value', fontsize=24)
    plt.tight_layout()
    # plt.savefig('resource/RuntimeUncertaintyCarRacing.pdf')
    plt.show()


if __name__ == "__main__":
    process_sweep("jjshoots/carracing_discrete/s73mfuy2", "CarRacing w/ Domain Change")
