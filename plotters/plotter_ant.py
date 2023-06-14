import logging
import warnings
from multiprocessing import Pool

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc, rcParams
from rliable import library as rly
from rliable import metrics, plot_utils

import wandb

# See warnings only once
warnings.filterwarnings("default")

# The answer to life, universe and everything
RAND_STATE = np.random.RandomState(42)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rc("text", usetex=False)
rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

sns.set_style("white")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def get_log_from_uri(uri, keys, api=None):
    assert isinstance(keys, list), "keys must be a list."
    api = wandb.Api(timeout=60) if api is None else api
    run = api.run(uri)
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        array = array.astype(np.float64)
        # array = np.nan_to_num(array, nan=-100.0, posinf=-100.0, neginf=-100.0)
        data[key] = array

    return data


def get_log_from_run(run, keys):
    assert isinstance(keys, list), "keys must be a list."
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        array = array.astype(np.float64)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        data[key] = array

    return data


def process_sweeps(title, sweep_uri_dict, baselines_dict):
    print(f"Processing run {title}")

    # parameters
    num_steps = 2000000
    num_intervals = 101
    color_palette = sns.color_palette("colorblind")

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # initialize API
    api = wandb.Api(timeout=30)

    # collect runs from sweeps
    runs = dict()
    for key in sweep_uri_dict:
        runs[key] = api.sweep(sweep_uri_dict[key]).runs

    # list of algorithms we have
    algorithms = [key for key in runs]

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    scores = dict()
    for algorithm in runs:
        score = []
        for run in runs[algorithm]:
            log = get_log_from_run(run, ["num_transitions", "eval_perf"])
            if log["num_transitions"].shape[0] > 80:
                data = np.interp(x_axis, log["num_transitions"], log["eval_perf"])
                score.append(data)

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
        custom_color=color_palette,
        algorithms=algorithms,
        xlabel=r"Timesteps (1e4)",
        ylabel="Evaluation IQM",
        labelsize=30,
        ticklabelsize=30,
        figsize=(9, 9),
    )

    # oracle policies
    offset = len(algorithms)
    for i, key in enumerate(baselines_dict):
        plt.axhline(
            y=baselines_dict[key],
            color=color_palette[offset + i],
            # color="black",
            linestyle="--",
        )
        algorithms.append(key)

    # form the legend
    legend_patches = [
        patches.Patch(color=color, alpha=0.75) for color in color_palette
    ]
    plt.legend(
        legend_patches,
        algorithms,
        loc="lower right",
        fancybox=True,
        # ncol=len(algorithms),
        ncol=2,
        fontsize=18,
        # handleheight=1.8,
        # bbox_to_anchor=(0.5, 1.4),
    )

    plt.title(
        title,
        fontsize=30,
    )
    plt.tight_layout()
    plt.savefig(f"resource/weak_ant.pdf", dpi=100)
    # plt.show()


if __name__ == "__main__":
    # list of run arguments
    sweep_objects = []

    title = "Ant-v4"
    sweep_uri_dict = {}
    sweep_uri_dict["SAC"] = "jjshoots/CCGE2/00dtckco"
    sweep_uri_dict["SAC_RR"] = "jjshoots/CCGE2/sc5wfw83"
    sweep_uri_dict["SAC_Ext"] = "jjshoots/CCGE2/vigk09s5"
    sweep_uri_dict["SAC_Ext_RR"] = "jjshoots/CCGE2/fp6vqi7q"
    sweep_uri_dict["CCGE"] = "jjshoots/CCGE2/i5l7a9wm"
    sweep_uri_dict["CCGE_RR"] = "jjshoots/CCGE2/3oeiia5t"
    sweep_uri_dict["CCGE_Ext"] = "jjshoots/CCGE2/g0nwlo72"
    sweep_uri_dict["CCGE_Ext_RR"] = "jjshoots/CCGE2/76pwgxu4"

    baselines_dict = {}
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    # process everything with multiprocessing
    with Pool() as pool:
        pool.starmap(process_sweeps, sweep_objects)