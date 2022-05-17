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
    num_intervals = 21

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
        ylabel="Evaluation Interquartile Mean (IQM)",
        labelsize="large",
        ticklabelsize="large",
    )

    # plot suboptimal policy lines
    for i, baseline in enumerate(baselines):
        plt.axhline(
            y=baselines[baseline],
            color=sns.color_palette("colorblind")[3 + i],
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
        "jjshoots/e2SAC_pybullet/27cjzt50",
        "jjshoots/e2SAC_pybullet/6vgj57fp",
        "jjshoots/e2SAC_pybullet/1l2a1jz3",
        "jjshoots/e2SAC_pybullet/110jrok2",
        "jjshoots/e2SAC_pybullet/1xyiilpo",
        "jjshoots/e2SAC_pybullet/2x5hkmbc",
        "jjshoots/e2SAC_pybullet/13r7w035",
        "jjshoots/e2SAC_pybullet/3pmbwojt",
        "jjshoots/e2SAC_pybullet/sj2wj9n9",
        "jjshoots/e2SAC_pybullet/12nev3lk",
    ]
    runs["CGES_1"] = [
        "jjshoots/pybullet_proper/1aexce2z",
        "jjshoots/pybullet_proper/1y6jhrec",
        "jjshoots/pybullet_proper/3k4nolub",
        "jjshoots/pybullet_proper/20yxnnax",
        "jjshoots/pybullet_proper/ngk7rmx1",
        "jjshoots/pybullet_proper/2z08ng6b",
        "jjshoots/pybullet_proper/30v3ld90",
        "jjshoots/pybullet_proper/1w6dc3wd",
        "jjshoots/pybullet_proper/22hd0shm",
        "jjshoots/pybullet_proper/387dg6td",
        "jjshoots/pybullet_proper/3bg0jygk",
        # "jjshoots/e2SAC_pybullet/26iy23z8",
        # "jjshoots/e2SAC_pybullet/z30or5e0",
        # "jjshoots/e2SAC_pybullet/2r6j4ti6",
        # "jjshoots/e2SAC_pybullet/1tahjde3",
        # "jjshoots/e2SAC_pybullet/8gef3h2b",
        # "jjshoots/e2SAC_pybullet/2ie3jxhp",
        # "jjshoots/e2SAC_pybullet/1936sb55",
        # "jjshoots/e2SAC_pybullet/3vq6jk5u",
        # "jjshoots/e2SAC_pybullet/28qlpnqd",
        # "jjshoots/e2SAC_pybullet/15h8pi1s",
    ]
    runs["CGES_2"] = [
        "jjshoots/pybullet_proper/twiy7oxs",
        "jjshoots/pybullet_proper/34vxuxlm",
        "jjshoots/pybullet_proper/3d08tjt0",
        "jjshoots/pybullet_proper/89b6d9xt",
        "jjshoots/pybullet_proper/1aexce2z",
        "jjshoots/pybullet_proper/2yccizor",
        "jjshoots/pybullet_proper/2vla6sxu",
        "jjshoots/pybullet_proper/3j0fq7rx",
        "jjshoots/pybullet_proper/tn3ubq2t",
        "jjshoots/pybullet_proper/u2jmu5sx",
    ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle_1"] = 1720.0
    baselines["Oracle_2"] = 2355.0
    baseline_list.append(baselines)
    env_list.append("AntPyBulletEnv-v0")

    # Hopper
    runs = {}
    runs["SAC"] = [
        "jjshoots/e2SAC_pybullet/o4vokgow",
        "jjshoots/e2SAC_pybullet/i1sfyc0a",
        "jjshoots/e2SAC_pybullet/2wvjd3l2",
        "jjshoots/e2SAC_pybullet/2g5l9qsb",
        "jjshoots/e2SAC_pybullet/3dfhch3p",
        "jjshoots/e2SAC_pybullet/3ourr309",
        "jjshoots/e2SAC_pybullet/29gbbi56",
        "jjshoots/e2SAC_pybullet/2x0lcpuj",
        "jjshoots/e2SAC_pybullet/3l6czfp3",
        "jjshoots/e2SAC_pybullet/2ciy5u3x",
    ]
    runs["CGES_1"] = [
        "jjshoots/pybullet_proper/3ruha6kc",
        "jjshoots/pybullet_proper/2ko2c2aj",
        "jjshoots/pybullet_proper/6h88uw0w",
        "jjshoots/pybullet_proper/ladktazj",
        "jjshoots/pybullet_proper/2y36fnp5",
        "jjshoots/pybullet_proper/2w2z9umd",
        "jjshoots/pybullet_proper/396k5lw4",
        "jjshoots/pybullet_proper/2nvsj0t8",
        "jjshoots/pybullet_proper/ml2uvnmw",
        "jjshoots/pybullet_proper/cmos8rof",
        # "jjshoots/e2SAC_pybullet/3v68x43e",
        # "jjshoots/e2SAC_pybullet/3dcmg6sb",
        # "jjshoots/e2SAC_pybullet/18c1nnmi",
        # "jjshoots/e2SAC_pybullet/3q62xhiu",
        # "jjshoots/e2SAC_pybullet/3tiax2q9",
        # "jjshoots/e2SAC_pybullet/1g6h6k8p",
        # "jjshoots/e2SAC_pybullet/2ihw442t",
        # "jjshoots/e2SAC_pybullet/brftgjqy",
        # "jjshoots/e2SAC_pybullet/2ttvam1p",
        # "jjshoots/e2SAC_pybullet/1cfh2ob3",
    ]
    runs["CGES_2"] = [
        "jjshoots/pybullet_proper/11rcgun1",
        "jjshoots/pybullet_proper/1u80mtnd",
        "jjshoots/pybullet_proper/1niisv0w",
        "jjshoots/pybullet_proper/kuxshzg0",
        "jjshoots/pybullet_proper/3n6guebt",
        "jjshoots/pybullet_proper/30qr6wr5",
        "jjshoots/pybullet_proper/3fz6ygp3",
        "jjshoots/pybullet_proper/1j2ugmdl",
        "jjshoots/pybullet_proper/1cwj8sav",
        "jjshoots/pybullet_proper/2phhovh3",
    ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle_1"] = 1677.0
    baselines["Oracle_2"] = 2500.0
    baseline_list.append(baselines)
    env_list.append("HopperPyBulletEnv-v0")

    # HalfCheetah
    runs = {}
    runs["SAC"] = [
        "jjshoots/e2SAC_pybullet/3t3pater",
        "jjshoots/e2SAC_pybullet/1im0kz8o",
        "jjshoots/e2SAC_pybullet/3p6rjq7d",
        "jjshoots/e2SAC_pybullet/3jjhmgb3",
        "jjshoots/e2SAC_pybullet/3sw13nvk",
        "jjshoots/e2SAC_pybullet/3671tc3j",
        "jjshoots/e2SAC_pybullet/3k7wa339",
        "jjshoots/e2SAC_pybullet/1gq9vj1w",
        "jjshoots/e2SAC_pybullet/1xla4kgx",
        "jjshoots/e2SAC_pybullet/3siwabhk",
    ]
    runs["CGES_1"] = [
        "jjshoots/pybullet_proper/pg1h5nds",
        "jjshoots/pybullet_proper/3jufwzxx",
        "jjshoots/pybullet_proper/2d51z0e4",
        "jjshoots/pybullet_proper/1zy4jl2u",
        "jjshoots/pybullet_proper/cug2h462",
        "jjshoots/pybullet_proper/1tq7rwb2",
        "jjshoots/pybullet_proper/3lqx12f5",
        "jjshoots/pybullet_proper/qtkjn0vz",
        "jjshoots/pybullet_proper/1r1h9r90",
        "jjshoots/pybullet_proper/31udjhbm",
        "jjshoots/pybullet_proper/32llu182",
        "jjshoots/pybullet_proper/2deg8r3u",
        # "jjshoots/e2SAC_pybullet/3s7i0gtm",
        # "jjshoots/e2SAC_pybullet/119e98d6",
        # "jjshoots/e2SAC_pybullet/3ncp4r5o",
        # "jjshoots/e2SAC_pybullet/2w5964ry",
        # "jjshoots/e2SAC_pybullet/2fns49hc",
        # "jjshoots/e2SAC_pybullet/1l1lbdzy",
        # "jjshoots/e2SAC_pybullet/27dnblfc",
        # "jjshoots/e2SAC_pybullet/a6i1t5al",
        # "jjshoots/e2SAC_pybullet/2e92ggz5",
        # "jjshoots/e2SAC_pybullet/35kzfwxw",
    ]
    runs["CGES_2"] = [
        "jjshoots/pybullet_proper/2di7pva6",
        "jjshoots/pybullet_proper/1bzcji2j",
        "jjshoots/pybullet_proper/1l62xg29",
        "jjshoots/pybullet_proper/p0lmd4xq",
        "jjshoots/pybullet_proper/2d51z0e4",
        "jjshoots/pybullet_proper/1ra0s53a",
        "jjshoots/pybullet_proper/1slyxy8d",
        "jjshoots/pybullet_proper/2q3imw5q",
        "jjshoots/pybullet_proper/227pvile",
        "jjshoots/pybullet_proper/1osdpzz4",
    ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle_1"] = 447.0
    baselines["Oracle_2"] = 1640.0
    baseline_list.append(baselines)
    env_list.append("HalfCheetahPyBulletEnv-v0")

    # Walker2D
    runs = {}
    runs["SAC"] = [
        "jjshoots/e2SAC_pybullet/2mz2a10g",
        "jjshoots/e2SAC_pybullet/hsl79yjd",
        "jjshoots/e2SAC_pybullet/3rwkyxk5",
        "jjshoots/e2SAC_pybullet/3pbhkx3e",
        "jjshoots/e2SAC_pybullet/3vqvthdp",
        "jjshoots/e2SAC_pybullet/24mthlrl",
        "jjshoots/e2SAC_pybullet/33gw34d7",
        "jjshoots/e2SAC_pybullet/34p7mfyc",
        "jjshoots/e2SAC_pybullet/1tukgzvw",
        "jjshoots/e2SAC_pybullet/1scupef7",
    ]
    runs["CGES_1"] = [
        "jjshoots/pybullet_proper/1wqvc3e7",
        "jjshoots/pybullet_proper/ohtwl01k",
        "jjshoots/pybullet_proper/11fr7qgh",
        "jjshoots/pybullet_proper/16l4njhw",
        "jjshoots/pybullet_proper/3u3l5qhc",
        "jjshoots/pybullet_proper/2p2wfpk6",
        "jjshoots/pybullet_proper/mpojmj1k",
        "jjshoots/pybullet_proper/z5n9vb5g",
        "jjshoots/pybullet_proper/57udwl5w",
        "jjshoots/pybullet_proper/3nrmugw3",
        "jjshoots/pybullet_proper/1kknkrdi",
        # "jjshoots/e2SAC_pybullet/qy3fj41l",
        # "jjshoots/e2SAC_pybullet/2r2ocxx3",
        # "jjshoots/e2SAC_pybullet/3k7ipmip",
        # "jjshoots/e2SAC_pybullet/8vfdgp62",
        # "jjshoots/e2SAC_pybullet/3rq3f05i",
        # "jjshoots/e2SAC_pybullet/g5dtjuev",
        # "jjshoots/e2SAC_pybullet/32g555xp",
        # "jjshoots/e2SAC_pybullet/2fioechx",
        # "jjshoots/e2SAC_pybullet/1b9n57ov",
        # "jjshoots/e2SAC_pybullet/2ax8d0rd",
    ]
    runs["CGES_2"] = [
        "jjshoots/pybullet_proper/odnzikh0",
        "jjshoots/pybullet_proper/19835kd8",
        "jjshoots/pybullet_proper/2bbwhvlz",
        "jjshoots/pybullet_proper/23fied0h",
        "jjshoots/pybullet_proper/1wqvc3e7",
        "jjshoots/pybullet_proper/2j78d803",
        "jjshoots/pybullet_proper/2yizj1sg",
        "jjshoots/pybullet_proper/2eh01ldd",
        "jjshoots/pybullet_proper/2gdynrc6",
        "jjshoots/pybullet_proper/1eo23yrb",
    ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle_1"] = 788.0
    baselines["Oracle_2"] = 1733.0
    baseline_list.append(baselines)
    env_list.append("Walker2DPyBulletEnv-v0")

    for runs, env_name, baselines in zip(run_list, env_list, baseline_list):
        compute_plots(runs, env_name, baselines)
