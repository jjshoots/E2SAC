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

    # get the sweep handler
    sweep = wandb.Api(timeout=30).sweep(sweep_uri)

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games); in our case it'll be (n x 1)
    pre_list = []
    post_list = []
    for run in sweep.runs:
        log = get_wandb_log(
            run, ["pre_switchup_uncertainty", "post_switchup_uncertainty"], start_val
        )
        if len(log["pre_switchup_uncertainty"]) > 0:
            log["pre_switchup_uncertainty"] = log["pre_switchup_uncertainty"][0]
            pre_list.append(log["pre_switchup_uncertainty"])

        if len(log["post_switchup_uncertainty"]) > 0:
            log["post_switchup_uncertainty"] = log["post_switchup_uncertainty"][0]
            post_list.append(log["post_switchup_uncertainty"])

    # expand along num_games axis
    pre_list = np.expand_dims(pre_list, axis=1)
    post_list = np.expand_dims(post_list, axis=1)
    print(len(pre_list))

    # put things in a dict
    uncer_list = {}
    uncer_list["pre"] = pre_list
    uncer_list["post"] = post_list

    aggregate_func = lambda x: np.array(
        [
            # metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            # metrics.aggregate_mean(x),
        ]
    )
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        uncer_list, aggregate_func, reps=50000
    )

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["IQM"],
        algorithms=None,
        xlabel="Mean Episodic F-value",
    )

    plt.title("CarRacing w/ Domain Change F-value", fontsize=24)
    # plt.tight_layout()
    # plt.savefig('resource/RuntimeUncertaintyCarRacing.pdf')
    plt.show()


if __name__ == "__main__":
    process_sweep("jjshoots/carracing_discrete/s73mfuy2", "CarRacing w/ Domain Change")
    process_sweep("jjshoots/carracing_discrete/mv3zhd3i", "CarRacing w/ Domain Change")
