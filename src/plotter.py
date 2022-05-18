# rliable google
import matplotlib.patches as patches

# plotting styles
import matplotlib.pyplot as plt

# normal imports
import numpy as np
import seaborn as sns
from matplotlib import rc, rcParams
from rliable import library as rly
from rliable import metrics, plot_utils

# to import stuff from wandb
import wandb

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


def get_wandb_log(run_uri, keys):
    assert isinstance(keys, list), "keys must be a list."
    api = wandb.Api(timeout=30)
    run = api.run(run_uri)
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        array = array.astype(np.float64)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        data[key] = array

    return data


def compute_plots(runs, env_name, baselines):
    # parameters
    num_steps = 1e6
    num_intervals = 51

    # x_axis values to plot against
    x_axis = np.linspace(0, num_steps, num_intervals)

    # list of algorithms we have
    algorithms = [key for key in runs]

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    scores = {}
    for algorithm in runs:
        score = []
        for run_uri in runs[algorithm]:
            log = get_wandb_log(run_uri, ["num_transitions", "eval_perf"])
            score.append(np.interp(x_axis, log["num_transitions"], log["eval_perf"]))

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
        x_axis / 1e6,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Timesteps (1e6)",
        ylabel="IQM Episodic Total Reward",
        labelsize="large",
        ticklabelsize="large",
    )

    # plot suboptimal policy lines
    for i, baseline in enumerate(baselines):
        plt.axhline(
            y=baselines[baseline],
            color=sns.color_palette("colorblind")[len(runs) + i],
            linestyle="-",
        )
        algorithms.append(baseline)

    # form the legend
    color_dict = dict(zip(algorithms, sns.color_palette("colorblind")))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in algorithms
    ]
    plt.legend(
        fake_patches,
        algorithms,
        loc="lower right",
        fancybox=True,
        # ncol=len(algorithms),
        fontsize="large",
        # bbox_to_anchor=(0.5, 1.1),
    )

    plt.savefig(f"resource/{env_name}.pdf")


if __name__ == "__main__":
    # list of algorithms and their corresponding uris
    run_list = []
    env_list = []
    baseline_list = []

    # Ant
    runs = {}
    runs["SAC"] = [
        "jjshoots/pybullet_proper2/5u53xvbr",
        "jjshoots/pybullet_proper2/my12ylhy",
        "jjshoots/pybullet_proper2/qs8fjteo",
        "jjshoots/pybullet_proper2/2nos2ruq",
        "jjshoots/pybullet_proper2/2n7flvkr",
        "jjshoots/pybullet_proper2/3f5d6i2z",
        "jjshoots/pybullet_proper2/3j02w4uw",
        "jjshoots/pybullet_proper2/1lwslgqt",
        "jjshoots/pybullet_proper2/c7giyxke",
        "jjshoots/pybullet_proper2/1t9xa0xq",
        "jjshoots/pybullet_proper2/1l4vcb9e",
        "jjshoots/pybullet_proper2/odteaj0s",
    ]
    # runs["CCGE1"] = [
    # ]
    # runs["CCGE2"] = [
    # ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle1"] = 1720.0
    baselines["Oracle2"] = 2355.0
    baseline_list.append(baselines)
    env_list.append("AntPyBulletEnv-v0")

    # Hopper
    runs = {}
    runs["SAC"] = [
        "jjshoots/pybullet_proper2/3agi889l",
        "jjshoots/pybullet_proper2/1cerdbg1",
        "jjshoots/pybullet_proper2/vwkcs9e8",
        "jjshoots/pybullet_proper2/1mxw7c76",
        "jjshoots/pybullet_proper2/27q7wd3j",
        "jjshoots/pybullet_proper2/gyvok2m6",
        "jjshoots/pybullet_proper2/12xo59aa",
        "jjshoots/pybullet_proper2/2jq2bia7",
        "jjshoots/pybullet_proper2/2tzvly3r",
        "jjshoots/pybullet_proper2/xozzppob",
        "jjshoots/pybullet_proper2/3sw9iv69",
        "jjshoots/pybullet_proper2/3nek6wsm",
    ]
    # runs["CCGE1"] = [
    # ]
    # runs["CCGE2"] = [
    # ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle1"] = 1677.0
    baselines["Oracle2"] = 2500.0
    baseline_list.append(baselines)
    env_list.append("HopperPyBulletEnv-v0")

    # HalfCheetah
    runs = {}
    runs["SAC"] = [
        "jjshoots/pybullet_proper2/29djrz60",
        "jjshoots/pybullet_proper2/yuhrc1m6",
        "jjshoots/pybullet_proper2/2s15roye",
        "jjshoots/pybullet_proper2/22tcgqrk",
        "jjshoots/pybullet_proper2/3ma8hltf",
        "jjshoots/pybullet_proper2/2lzlligf",
        "jjshoots/pybullet_proper2/sc3pvis4",
        "jjshoots/pybullet_proper2/2eylgcg0",
        "jjshoots/pybullet_proper2/2bj2myc8",
        "jjshoots/pybullet_proper2/27uegbvt",
        "jjshoots/pybullet_proper2/2u1vsnr5",
        "jjshoots/pybullet_proper2/cel76d0i",
    ]
    # runs["CCGE1"] = [
    # ]
    # runs["CCGE2"] = [
    # ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle1"] = 447.0
    baselines["Oracle2"] = 1640.0
    baseline_list.append(baselines)
    env_list.append("HalfCheetahPyBulletEnv-v0")

    # Walker2D
    runs = {}
    runs["SAC"] = [
        "jjshoots/pybullet_proper2/iv83lvpe",
        "jjshoots/pybullet_proper2/1nugaeda",
        "jjshoots/pybullet_proper2/1l99vuyl",
        "jjshoots/pybullet_proper2/lgab8igk",
        "jjshoots/pybullet_proper2/xwziupc5",
        "jjshoots/pybullet_proper2/3d59105o",
        "jjshoots/pybullet_proper2/13p6apja",
        "jjshoots/pybullet_proper2/3bb8whna",
        "jjshoots/pybullet_proper2/1gsqjuow",
        "jjshoots/pybullet_proper2/1vyczwt2",
        "jjshoots/pybullet_proper2/1nd3v4dh",
        "jjshoots/pybullet_proper2/2000jz2c",
    ]
    # runs["CCGE1"] = [
    # ]
    # runs["CCGE2"] = [
    # ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle1"] = 788.0
    baselines["Oracle2"] = 1733.0
    baseline_list.append(baselines)
    env_list.append("Walker2DPyBulletEnv-v0")

    for runs, env_name, baselines in zip(run_list, env_list, baseline_list):
        compute_plots(runs, env_name, baselines)
