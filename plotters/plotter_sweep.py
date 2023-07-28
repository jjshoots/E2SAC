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
    num_steps = 1000000
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
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm, reps=100000)

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e4,
        iqm_scores,
        iqm_cis,
        custom_colors=color_palette,
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
    legend_patches = [patches.Patch(color=color, alpha=0.75) for color in color_palette]
    plt.legend(
        legend_patches,
        algorithms,
        loc="lower right",
        fancybox=True,
        # ncol=len(algorithms),
        ncol=1,
        fontsize=18,
        # handleheight=1.8,
        # bbox_to_anchor=(0.5, 1.4),
    )

    plt.title(
        title,
        fontsize=30,
    )
    plt.tight_layout()
    plt.savefig(f"resource/{title}.pdf", dpi=100)
    # plt.show()


if __name__ == "__main__":
    # list of run arguments
    sweep_objects = []

    title = "Hopper-v4"
    sweep_uri_dict = {}
    sweep_uri_dict["SAC"] = "jjshoots/CCGE2/zgq81g05"
    sweep_uri_dict["CCGE_1_Im"] = "jjshoots/CCGE2/8tex48bv"
    sweep_uri_dict["CCGE_1_Ex"] = "jjshoots/CCGE2/phevs4mc"
    sweep_uri_dict["CCGE_2_Im"] = "jjshoots/CCGE2/18p6t19p"
    sweep_uri_dict["CCGE_2_Ex"] = "jjshoots/CCGE2/po10jfqp"
    sweep_uri_dict["CCGE_B_Ex"] = "jjshoots/CCGE2/fscr3ka0"

    baselines_dict = {}
    baselines_dict["Oracle 1"] = 850.0
    baselines_dict["Oracle 2"] = 2900.0
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    title = "Ant-v4"
    sweep_uri_dict = {}
    sweep_uri_dict["SAC"] = "jjshoots/CCGE2/4uvx5qez"
    sweep_uri_dict["CCGE_1_Im"] = "jjshoots/CCGE2/d74a6t1w"
    sweep_uri_dict["CCGE_1_Ex"] = "jjshoots/CCGE2/dp4byhg8"
    sweep_uri_dict["CCGE_2_Im"] = "jjshoots/CCGE2/f9sntmmm"
    sweep_uri_dict["CCGE_2_Ex"] = "jjshoots/CCGE2/b0r4xjqu"
    sweep_uri_dict["CCGE_B_Ex"] = "jjshoots/CCGE2/5ydeyszy"

    baselines_dict = {}
    baselines_dict["Oracle 1"] = 980.0
    baselines_dict["Oracle 2"] = 2200.0
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    title = "HalfCheetah-v4"
    sweep_uri_dict = {}
    sweep_uri_dict["SAC"] = "jjshoots/CCGE2/95qoyb9w"
    sweep_uri_dict["CCGE_1_Im"] = "jjshoots/CCGE2/i9cg5mqw"
    sweep_uri_dict["CCGE_1_Ex"] = "jjshoots/CCGE2/o61ibecs"
    sweep_uri_dict["CCGE_2_Im"] = "jjshoots/CCGE2/5e3vqsse"
    sweep_uri_dict["CCGE_2_Ex"] = "jjshoots/CCGE2/62jvni7q"
    sweep_uri_dict["CCGE_B_Ex"] = "jjshoots/CCGE2/3rk3ayzb"

    baselines_dict = {}
    baselines_dict["Oracle 1"] = 5400.0
    baselines_dict["Oracle 2"] = 6000.0
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    title = "Walker2d-v4"
    sweep_uri_dict = {}
    sweep_uri_dict["SAC"] = "jjshoots/CCGE2/aboikqup"
    sweep_uri_dict["CCGE_1_Im"] = "jjshoots/CCGE2/trgcinhi"
    sweep_uri_dict["CCGE_1_Ex"] = "jjshoots/CCGE2/s34yh4a6"
    sweep_uri_dict["CCGE_2_Im"] = "jjshoots/CCGE2/nzkwl5p3"
    sweep_uri_dict["CCGE_2_Ex"] = "jjshoots/CCGE2/n6ufn853"
    sweep_uri_dict["CCGE_B_Ex"] = "jjshoots/CCGE2/44l7ajlp"

    baselines_dict = {}
    baselines_dict["Oracle 1"] = 2300.0
    baselines_dict["Oracle 2"] = 3100.0
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    # process everything with multiprocessing
    with Pool() as pool:
        pool.starmap(process_sweeps, sweep_objects)
