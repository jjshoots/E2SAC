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


def process_sweep(sweep_name, sweep_uri, num_steps, start_val, num_intervals=200):
    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms and their corresponding uris from sweep
    sweep = wandb.Api(timeout=30).sweep(sweep_uri)

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
    iqm = lambda eval_scores: np.array(
        [
            metrics.aggregate_iqm(eval_scores[..., frame])
            for frame in range(eval_scores.shape[-1])
        ]
    )
    eval_iqm, eval_cis = rly.get_interval_estimates(eval_scores, iqm, reps=50000)

    # get interquartile mean
    iqm = lambda uncertainties: np.array(
        [
            metrics.aggregate_iqm(uncertainties[..., frame])
            for frame in range(uncertainties.shape[-1])
        ]
    )
    uncer_iqm, uncer_cis = rly.get_interval_estimates(uncertainties, iqm, reps=50000)

    # instantiate colorwheel
    palette = sns.color_palette("colorblind")

    # twin plots
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel("Timestep (1e6)")
    ax1.tick_params(axis="x", labelsize=15)

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e6,
        eval_iqm,
        eval_cis,
        algorithms=None,
        xlabel=r"Timesteps (1e6)",
        ylabel="Evaluation Interquartile Mean (IQM)",
        labelsize=30,
        ticklabelsize=30,
        ax=ax1,
        custom_color=sns.color_palette("colorblind")[0:],
    )
    ax1.tick_params(axis="y", labelcolor=palette[0], labelsize=15)
    ax1.set_ylabel("Evaluation Interquartile Mean (IQM)", color=palette[0], fontsize=30)

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        x_axis / 1e6,
        uncer_iqm,
        uncer_cis,
        algorithms=None,
        xlabel=r"Timesteps (1e6)",
        ylabel="Episodic Mean Epistemic Uncertainty",
        labelsize=30,
        ticklabelsize=30,
        ax=ax2,
        custom_color=sns.color_palette("colorblind")[1:],
    )
    ax2.tick_params(axis="y", labelcolor=palette[1], labelsize=15)
    ax2.set_ylabel("Episodic Mean Epistemic Uncertainty", color=palette[1], fontsize=30)
    # ax2.set_ylim(top=2.0)

    plt.title(sweep_name, fontsize=30)
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(f"./resource/{sweep_name}.pdf")


if __name__ == "__main__":
    # sweeps["LunarLander100k_long"] = ["jjshoots/DQN2/dotzndpe", 3e6, -200.0]

    sweeps = {}
    sweeps["CartPole50k"] = ["jjshoots/DQN2/a0gjbznv", 0.25e6, 100.0]
    sweeps["CartPole100k"] = ["jjshoots/DQN2/u7k2k7qo", 0.25e6, 100.0]
    sweeps["CartPole200k"] = ["jjshoots/DQN2/emhvyijs", 0.25e6, 100.0]
    sweeps["Acrobot50k"] = ["jjshoots/DQN2/5bv1o5du", 0.25e6, -500.0]
    sweeps["Acrobot100k"] = ["jjshoots/DQN2/t3e9smkh", 0.25e6, -500.0]
    sweeps["Acrobot200k"] = ["jjshoots/DQN2/6ssn48ak", 0.25e6, -500.0]
    sweeps["MountainCar100k"] = ["jjshoots/DQN2/xy1blq0i", 1e6, -200.0]
    sweeps["MountainCar200k"] = ["jjshoots/DQN2/zlqur3uh", 1e6, -200.0]
    sweeps["MountainCar400k"] = ["jjshoots/DQN2/4zjlky9u", 1e6, -200.0]
    sweeps["LunarLander100k"] = ["jjshoots/DQN2/ns2i31ul", 1e6, -200.0]
    sweeps["LunarLander200k"] = ["jjshoots/DQN2/146u4rcg", 1e6, -200.0]
    sweeps["LunarLander400k"] = ["jjshoots/DQN2/0d1c22d0", 1e6, -200.0]

    for key in sweeps:
        process_sweep(key, sweeps[key][0], sweeps[key][1], sweeps[key][2])
