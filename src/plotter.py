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


def compute_plots(runs, env_name, baseline):
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

    # plot suboptimal policy line
    plt.axhline(y=baseline, color=sns.color_palette("colorblind")[2], linestyle="-")
    algorithms.append("Suboptimal")

    # form the legend
    color_dict = dict(zip(algorithms, sns.color_palette("colorblind")))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in algorithms
    ]
    plt.legend(
        fake_patches,
        algorithms,
        loc="upper center",
        fancybox=True,
        ncol=len(algorithms),
        fontsize="large",
        bbox_to_anchor=(0.5, 1.1),
    )

    plt.savefig(f"resource/{env_name}.pdf")


if __name__ == "__main__":
    # list of algorithms and their corresponding uris
    run_list = []
    env_list = []
    baselines = []

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
    runs["E2SAC"] = [
        # "jjshoots/pybullet_proper/3bg0jygk",
        "jjshoots/pybullet_proper/1w6dc3wd",
        "jjshoots/pybullet_proper/22hd0shm",
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
    run_list.append(runs)
    env_list.append("AntPyBulletEnv-v0")
    baselines.append(1720.0)

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
    runs["E2SAC"] = [
        "jjshoots/pybullet_proper/cmos8rof",
        "jjshoots/pybullet_proper/396k5lw4",
        "jjshoots/pybullet_proper/2nvsj0t8",
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
    run_list.append(runs)
    env_list.append("HopperPyBulletEnv-v0")
    baselines.append(1677.0)

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
    runs["E2SAC"] = [
        # "jjshoots/pybullet_proper/2deg8r3u",
        "jjshoots/pybullet_proper/31udjhbm",
        "jjshoots/pybullet_proper/1r1h9r90",
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
    run_list.append(runs)
    env_list.append("HalfCheetahPyBulletEnv-v0")
    baselines.append(447.0)

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
    runs["E2SAC"] = [
        # "jjshoots/pybullet_proper/1kknkrdi",
        "jjshoots/pybullet_proper/57udwl5w",
        "jjshoots/pybullet_proper/z5n9vb5g",
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
    run_list.append(runs)
    env_list.append("Walker2DPyBulletEnv-v0")
    baselines.append(788.0)

    for runs, env_name, baseline in zip(run_list, env_list, baselines):
        compute_plots(runs, env_name, baseline)
