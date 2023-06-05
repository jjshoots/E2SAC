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


def process_sweeps(title, sweep_uri):
    # collect runs from sweeps
    api = wandb.Api(timeout=30)
    runs = api.sweep(sweep_uri).runs

    # the x axis
    x_vals = [run.config["confidence_lambda"] for run in runs]

    # sup scale metrics
    sup_scale = []
    for run in runs:
        history = run.scan_history(["sup_scale"])
        sup_scale.append(np.array([row["sup_scale"] for row in history]))
    sup_scale_mean = [x.mean() for x in sup_scale]
    sup_scale_var = [x.var() for x in sup_scale]

    plt.scatter(x_vals, sup_scale_mean)
    plt.scatter(x_vals, sup_scale_var)
    plt.show()

    # eval metrics
    eval_perf = []
    for run in runs:
        history = run.scan_history(["eval_perf"])
        eval_perf.append(np.array([row["eval_perf"] for row in history]))
    eval_perf_max = [x.max() for x in eval_perf]
    eval_perf_mean = [x.mean() for x in eval_perf]
    eval_perf_var = [x.var() for x in eval_perf]

    plt.close()
    plt.scatter(x_vals, eval_perf_max)
    plt.scatter(x_vals, eval_perf_mean)
    plt.show()

if __name__ == "__main__":
    title = "lambdaSensitivity:PyFlyt"
    sweep_uri = "jjshoots/CCGE2/k1vw5iq6"

    process_sweeps(title, sweep_uri)
