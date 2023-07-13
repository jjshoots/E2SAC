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
    api = wandb.Api(timeout=30) if api is None else api
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
    num_steps = 1e6
    num_intervals = 100

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
            if log["num_transitions"].shape[0] > 60:
                # exception if num_transitions doesn't start at 10k
                if log["num_transitions"][0] != 10000:
                    start_transitions = [
                        i for i in range(10000, int(log["num_transitions"][0]))
                    ]
                    start_eval_perf = [0.0] * len(start_transitions)

                    log["num_transitions"] = np.concatenate(
                        [start_transitions, log["num_transitions"]], axis=-1
                    )
                    log["eval_perf"] = np.concatenate(
                        [start_eval_perf, log["eval_perf"]], axis=-1
                    )

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
        algorithms=algorithms,
        xlabel=r"Timesteps (1e4)",
        ylabel="Evaluation Waypoints Reached",
        labelsize=30,
        ticklabelsize=30,
        figsize=(9, 9),
    )

    color_palette = sns.color_palette("colorblind")

    # oracle policies
    for i, key in enumerate(baselines_dict):
        line = plt.axhline(
            y=baselines_dict[key],
            color=color_palette[len(algorithms) + i],
            # color="black",
            linestyle="--",
        )
        algorithms.append(key)

    # form the legend
    color_dict = dict(zip(algorithms, color_palette))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in algorithms
    ]
    legend = plt.legend(
        fake_patches,
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
        title.replace(":", "/"),
        fontsize=30,
    )
    plt.tight_layout()
    filename = title.split("\n")[0]
    plt.savefig(f"resource/{filename}-guide-sup.pdf", dpi=100)
    # plt.show()


if __name__ == "__main__":
    # list of run arguments
    sweep_objects = []

    title = "Walker2d-v4\nGuidance vs. Supervision"
    sweep_uri_dict = {}
    sweep_uri_dict["Supervision + Guidance"] = "jjshoots/CCGE2/n6ufn853"
    sweep_uri_dict["Guidance Only"] = "jjshoots/CCGE2/3a92kbas"
    sweep_uri_dict["Supervision Only"] = "jjshoots/CCGE2/wtfin9dd"

    baselines_dict = {}
    baselines_dict["Oracle"] = 3100.0
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    title = "Ant-v4\nGuidance vs. Supervision"
    sweep_uri_dict = {}
    sweep_uri_dict["Supervision + Guidance"] = "jjshoots/CCGE2/b0r4xjqu"
    sweep_uri_dict["Guidance Only"] = "jjshoots/CCGE2/3d8yleyc"
    sweep_uri_dict["Supervision Only"] = "jjshoots/CCGE2/vxtb4692"

    baselines_dict = {}
    baselines_dict["Oracle"] = 2200.0
    sweep_objects.append((title, sweep_uri_dict, baselines_dict))

    # process everything with multiprocessing
    with Pool() as pool:
        pool.starmap(process_sweeps, sweep_objects)
