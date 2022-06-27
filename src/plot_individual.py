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


def process_run(run_name, run_uri, num_steps, start_val, num_intervals=200):
    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris from sweep
    run = wandb.Api(timeout=30).run(run_uri)

    # get the run items we want
    log = get_wandb_log(
        run, ["num_transitions", "eval_perf", "runtime_uncertainty"], start_val
    )
    eval = np.interp(x_axis, log["num_transitions"], log["eval_perf"])
    uncer = np.interp(x_axis, log["num_transitions"], log["runtime_uncertainty"])

    # instantiate colorwheel
    palette = sns.color_palette("colorblind")

    # twin plots
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Timestep (1e6)", fontsize=15)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.plot(x_axis / 1e6, eval, color=palette[0])

    ax1.tick_params(axis="y", labelcolor=palette[0], labelsize=15)
    ax1.set_ylabel("Evaluation Score", color=palette[0], fontsize=20)
    ax2.plot(x_axis / 1e6, uncer, color=palette[1])

    ax2.tick_params(axis="y", labelcolor=palette[1], labelsize=15)
    ax2.set_ylabel("Episodic Mean F-value", color=palette[1], fontsize=20)
    # ax2.set_ylim(top=2.0)

    plt.title(run_name, fontsize=20)


if __name__ == "__main__":

    runs = {}
    runs["LunarLander_1"] = ["jjshoots/DQN2/v7m9fdid", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_2"] = ["jjshoots/DQN2/0nryzwe0", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_3"] = ["jjshoots/DQN2/9a87kzrw", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_4"] = ["jjshoots/DQN2/4to08c08", 1e6, -300.0, "LunarLander"]
    runs["MountainCar_1"] = ["jjshoots/DQN2/rl7h863y", 1e6, -200.0, "MountainCar"]
    runs["MountainCar_2"] = ["jjshoots/DQN2/fao9ib3w", 1e6, -200.0, "MountainCar"]
    runs["MountainCar_3"] = ["jjshoots/DQN2/teotgmwk", 1e6, -200.0, "MountainCar"]
    runs["MountainCar_4"] = ["jjshoots/DQN2/5kx1l41n", 1e6, -200.0, "MountainCar"]

    for key in runs:
        process_run(runs[key][-1], runs[key][0], runs[key][1], runs[key][2])

    plt.show()
