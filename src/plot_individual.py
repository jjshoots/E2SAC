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


def process_run(run_name, env_name, run_uri, num_steps, start_val, num_intervals=200):
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
    ax1.set_xlabel("Timestep (1e6)", fontsize=30)
    ax1.tick_params(axis="x", labelsize=30, length=0)
    ax1.plot(x_axis / 1e6, eval, color=palette[0])

    ax1.tick_params(axis="y", labelcolor=palette[0], labelsize=30, length=0)
    ax1.set_ylabel("Evaluation Score", color=palette[0], fontsize=30)
    ax2.plot(x_axis / 1e6, uncer, color=palette[1])

    ax2.tick_params(axis="y", labelcolor=palette[1], labelsize=30, length=0)
    ax2.set_ylabel("Episodic Mean F-value", color=palette[1], fontsize=30)
    # ax2.set_ylim(top=2.0)

    # remove the spines for top and right
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)

    plt.title(env_name, fontsize=30)
    fig.set_size_inches(8, 8)
    plt.tight_layout()
    plt.savefig(f"./resource/{run_name}.pdf")


if __name__ == "__main__":

    runs = {}
    # runs["CartPole_1"] = ["jjshoots/DQN2/t3s6ynwy", 250e3, -0.0, "CartPole"]
    # runs["CartPole_2"] = ["jjshoots/DQN2/xhlma4bk", 250e3, -0.0, "CartPole"]
    # runs["CartPole_3"] = ["jjshoots/DQN2/umi2l2cs", 250e3, -0.0, "CartPole"]
    # runs["CartPole_4"] = ["jjshoots/DQN2/k7qop4kd", 250e3, -0.0, "CartPole"]
    # runs["CartPole_5"] = ["jjshoots/DQN2/yxcah00a", 250e3, -0.0, "CartPole"]
    # runs["CartPole_6"] = ["jjshoots/DQN2/90rsze3y", 250e3, -0.0, "CartPole"]
    # runs["CartPole_7"] = ["jjshoots/DQN2/46mz3dt5", 250e3, -0.0, "CartPole"]
    # runs["CartPole_8"] = ["jjshoots/DQN2/9m26fb92", 250e3, -0.0, "CartPole"]
    # runs["CartPole_9"] = ["jjshoots/DQN2/w05zpas0", 250e3, -0.0, "CartPole"]
    # runs["CartPole_10"] = ["jjshoots/DQN2/3cfd0vli", 250e3, -0.0, "CartPole"]
    # runs["CartPole_11"] = ["jjshoots/DQN2/a0lmll1v", 250e3, -0.0, "CartPole"]
    # runs["CartPole_12"] = ["jjshoots/DQN2/3391a5ns", 250e3, -0.0, "CartPole"]
    # runs["Acrobot_1"] = ["jjshoots/DQN2/8vls2940", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_2"] = ["jjshoots/DQN2/y1uo0wqf", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_3"] = ["jjshoots/DQN2/yzgoqj14", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_4"] = ["jjshoots/DQN2/og0kdavv", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_5"] = ["jjshoots/DQN2/06627ljd", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_6"] = ["jjshoots/DQN2/pk6zypcl", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_7"] = ["jjshoots/DQN2/ttc68q0o", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_8"] = ["jjshoots/DQN2/f598uymf", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_9"] = ["jjshoots/DQN2/ejysien9", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_10"] = ["jjshoots/DQN2/ednfew92", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_11"] = ["jjshoots/DQN2/1doxpo8i", 250e3, -500.0, "Acrobot"]
    # runs["Acrobot_12"] = ["jjshoots/DQN2/3ih3yarp", 250e3, -500.0, "Acrobot"]
    runs["LunarLander_1"] = ["jjshoots/DQN2/v7m9fdid", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_2"] = ["jjshoots/DQN2/0nryzwe0", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_3"] = ["jjshoots/DQN2/9a87kzrw", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_4"] = ["jjshoots/DQN2/4to08c08", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_5"] = ["jjshoots/DQN2/zjxgzu6x", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_6"] = ["jjshoots/DQN2/epvt9tjx", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_7"] = ["jjshoots/DQN2/pwolbjqa", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_8"] = ["jjshoots/DQN2/784j0p5b", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_9"] = ["jjshoots/DQN2/5sctvm2p", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_10"] = ["jjshoots/DQN2/0nryzwe0", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_11"] = ["jjshoots/DQN2/23a1eenq", 1e6, -300.0, "LunarLander"]
    runs["LunarLander_12"] = ["jjshoots/DQN2/ee2oc03o", 1e6, -300.0, "LunarLander"]
    # runs["MountainCar_1"] = ["jjshoots/DQN2/rl7h863y", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_2"] = ["jjshoots/DQN2/fao9ib3w", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_3"] = ["jjshoots/DQN2/teotgmwk", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_4"] = ["jjshoots/DQN2/5kx1l41n", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_5"] = ["jjshoots/DQN2/lj2psabh", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_6"] = ["jjshoots/DQN2/ltya08ps", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_7"] = ["jjshoots/DQN2/92ksv6xq", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_8"] = ["jjshoots/DQN2/y1bc61xw", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_9"] = ["jjshoots/DQN2/30baols3", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_10"] = ["jjshoots/DQN2/a1rg2vby", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_11"] = ["jjshoots/DQN2/8cou63if", 1e6, -200.0, "MountainCar"]
    # runs["MountainCar_12"] = ["jjshoots/DQN2/km2jue7g", 1e6, -200.0, "MountainCar"]

    for key in runs:
        process_run(key, runs[key][-1], runs[key][0], runs[key][1], runs[key][2])