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

split_size = 8


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

    # sup scale metrics
    # this is an array of [x_vals, 10] for each 100k
    x_vals = []
    sup_ratios = []
    eval_perfs = []
    for run in runs:
        history = run.scan_history(["sup_scale", "eval_perf"])
        if history.max_step < 200:
            continue

        x_vals.append(run.config["confidence_lambda"])
        history = run.scan_history(["sup_scale", "eval_perf"])

        # record the supervision ratios
        log = np.array([row["sup_scale"] for row in history])
        log = log.reshape(split_size, -1)
        log = log.mean(axis=-1)
        sup_ratios.append(log)

        log = np.array([row["eval_perf"] for row in history])
        log = log.reshape(split_size, -1)
        log = log.mean(axis=-1)
        eval_perfs.append(log)

    # sort the things
    # the x axis, sorted
    sorted_indices = np.argsort(x_vals)
    x_vals = np.take(x_vals, sorted_indices)
    sup_ratios = np.take(sup_ratios, sorted_indices, axis=0)
    eval_perfs = np.take(eval_perfs, sorted_indices, axis=0)

    # save the data
    np.save("./sensitivity_data/x_vals.npy", x_vals)
    np.save("./sensitivity_data/sup_ratios.npy", sup_ratios)
    np.save("./sensitivity_data/eval_perfs.npy", eval_perfs)


def plot_data():
    x_vals = np.load("./sensitivity_data/x_vals.npy")
    sup_ratios = np.load("./sensitivity_data/sup_ratios.npy")
    eval_perfs = np.load("./sensitivity_data/eval_perfs.npy")

    # kernel for data smoothing
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    # colors
    # color_palette = sns.color_palette("colorblind")
    color_palette = sns.color_palette("Reds", n_colors=8)

    # interpolation range
    x_axis = np.linspace(0, 5.0, 200)

    # all the labels
    labels = ["0-250k", "250-500k", "500-750k", "750-1000k", "1000-1250k", "1250-1500k", "1500-1750k", "1750-2000k"]

    for i, vals in enumerate(sup_ratios.T):
        vals = np.interp(x_axis, x_vals, vals)
        vals = np.convolve(vals, kernel, mode="valid")
        plt.plot(x_axis[:len(vals)], vals, c=color_palette[i], label=labels[i])
    plt.title("Mean Supervision Ratio \nvs. Confidence Scale", fontsize=30)
    # plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig("sensitivity_data/sup_vs_conf.pdf", dpi=100)
    plt.close()

    for i, vals in enumerate(eval_perfs.T):
        vals = np.interp(x_axis, x_vals, vals)
        vals = np.convolve(vals, kernel, mode="valid")
        plt.plot(x_axis[:len(vals)], vals, c=color_palette[i], label=labels[i])
    plt.title("Mean Evaluation Performance \nvs. Confidence Scale", fontsize=30)
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"sensitivity_data/eval_vs_conf.pdf", dpi=100)
    plt.close()


if __name__ == "__main__":
    title = "lambdaSensitivity:Ant_v4"
    sweep_uri = "jjshoots/CCGE2/oqxdqmhk"

    # process_sweeps(title, sweep_uri)
    plot_data()
