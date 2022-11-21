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
    api = wandb.Api(timeout=180)
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
        labelsize=24,
        ticklabelsize=24,
        figsize=(9, 9),
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
        loc="upper left",
        fancybox=True,
        # ncol=len(algorithms),
        fontsize=18,
        # bbox_to_anchor=(0.5, 1.1),
    )

    plt.title(env_name, fontsize=24)
    plt.tight_layout()
    plt.savefig(f"resource/{env_name}.pdf")


if __name__ == "__main__":
    # list of algorithms and their corresponding uris
    run_list = []
    env_list = []
    baseline_list = []

    # # Ant
    # runs = {}
    # runs["SAC"] = [
    #     "jjshoots/pybullet_proper2/5u53xvbr",
    #     "jjshoots/pybullet_proper2/my12ylhy",
    #     "jjshoots/pybullet_proper2/qs8fjteo",
    #     "jjshoots/pybullet_proper2/2nos2ruq",
    #     "jjshoots/pybullet_proper2/2n7flvkr",
    #     "jjshoots/pybullet_proper2/3f5d6i2z",
    #     "jjshoots/pybullet_proper2/3j02w4uw",
    #     "jjshoots/pybullet_proper2/1lwslgqt",
    #     "jjshoots/pybullet_proper2/c7giyxke",
    #     "jjshoots/pybullet_proper2/1t9xa0xq",
    #     "jjshoots/pybullet_proper2/1l4vcb9e",
    #     "jjshoots/pybullet_proper2/odteaj0s",
    # ]
    # runs["CCGE no bias w/ Oracle 2"] = [
    #     "jjshoots/pybullet_proper2/47vas0et",
    #     "jjshoots/pybullet_proper2/2ud5d7io",
    #     "jjshoots/pybullet_proper2/1hgvkjsz",
    #     "jjshoots/pybullet_proper2/19x23xod",
    #     "jjshoots/pybullet_proper2/1u90gdof",
    #     "jjshoots/pybullet_proper2/3jr4fw5t",
    #     "jjshoots/pybullet_proper2/1y6ylnmh",
    #     "jjshoots/pybullet_proper2/1gmnwa4g",
    #     "jjshoots/pybullet_proper2/24c90bcd",
    #     "jjshoots/pybullet_proper2/243z7v5o",
    # ]
    # runs["CCGE bias w/ Oracle 2"] = [
    #     "jjshoots/pybullet_proper2/1a86ublm",
    #     "jjshoots/pybullet_proper2/3ka9ihez",
    #     "jjshoots/pybullet_proper2/262wf2zo",
    #     "jjshoots/pybullet_proper2/1lq7pqxm",
    #     "jjshoots/pybullet_proper2/154uq9o3",
    #     "jjshoots/pybullet_proper2/21gzao6p",
    #     "jjshoots/pybullet_proper2/1da57v6a",
    #     "jjshoots/pybullet_proper2/1lcobiai",
    #     "jjshoots/pybullet_proper2/3uerg3af",
    #     "jjshoots/pybullet_proper2/2jp1jwwq",
    # ]
    # runs["CCGE no bias w/ Oracle 1"] = [
    #     "jjshoots/pybullet_proper2/hv9k0u9z",
    #     "jjshoots/pybullet_proper2/32akryk0",
    #     "jjshoots/pybullet_proper2/ql7i68yy",
    #     "jjshoots/pybullet_proper2/2i76if75",
    #     "jjshoots/pybullet_proper2/1nlbjy47",
    #     "jjshoots/pybullet_proper2/1fxk7zpi",
    #     "jjshoots/pybullet_proper2/trnzurhq",
    #     "jjshoots/pybullet_proper2/166eludf",
    #     "jjshoots/pybullet_proper2/18faa4fy",
    #     "jjshoots/pybullet_proper2/2b2mn47g",
    # ]
    # run_list.append(runs)
    # baselines = {}
    # baselines["Oracle 1"] = 1074.0
    # baselines["Oracle 2"] = 2107.0
    # baseline_list.append(baselines)
    # env_list.append("AntPyBulletEnv-v0")

    # # Hopper
    # runs = {}
    # runs["SAC"] = [
    #     "jjshoots/pybullet_proper2/3agi889l",
    #     "jjshoots/pybullet_proper2/1cerdbg1",
    #     "jjshoots/pybullet_proper2/vwkcs9e8",
    #     "jjshoots/pybullet_proper2/1mxw7c76",
    #     "jjshoots/pybullet_proper2/27q7wd3j",
    #     "jjshoots/pybullet_proper2/gyvok2m6",
    #     "jjshoots/pybullet_proper2/12xo59aa",
    #     "jjshoots/pybullet_proper2/2jq2bia7",
    #     "jjshoots/pybullet_proper2/2tzvly3r",
    #     "jjshoots/pybullet_proper2/xozzppob",
    #     "jjshoots/pybullet_proper2/3sw9iv69",
    #     "jjshoots/pybullet_proper2/3nek6wsm",
    # ]
    # runs["CCGE no bias w/ Oracle 2"] = [
    #     "jjshoots/pybullet_proper2/31m1ydp4",
    #     "jjshoots/pybullet_proper2/bt5tcn2u",
    #     "jjshoots/pybullet_proper2/1l5of0xc",
    #     "jjshoots/pybullet_proper2/771eb2ye",
    #     "jjshoots/pybullet_proper2/266spyl2",
    #     "jjshoots/pybullet_proper2/1xcwyouz",
    #     "jjshoots/pybullet_proper2/2iz9ynna",
    #     "jjshoots/pybullet_proper2/2k95xw45",
    #     "jjshoots/pybullet_proper2/2asykiz1",
    #     "jjshoots/pybullet_proper2/lvahv041",
    # ]
    # runs["CCGE bias w/ Oracle 2"] = [
    #     "jjshoots/pybullet_proper2/1nri49lb",
    #     "jjshoots/pybullet_proper2/3u0bafwn",
    #     "jjshoots/pybullet_proper2/33bf7biz",
    #     "jjshoots/pybullet_proper2/1mq8gj6i",
    #     "jjshoots/pybullet_proper2/saakuj9v",
    #     "jjshoots/pybullet_proper2/r2elxfv6",
    #     "jjshoots/pybullet_proper2/2ki1ksjk",
    #     "jjshoots/pybullet_proper2/1fj8pus4",
    #     "jjshoots/pybullet_proper2/2nyxbg6b",
    # ]
    # runs["CCGE no bias w/ Oracle 1"] = [
    #     "jjshoots/pybullet_proper2/1ys86489",
    #     "jjshoots/pybullet_proper2/ozn1uhyp",
    #     "jjshoots/pybullet_proper2/1limtj36",
    #     "jjshoots/pybullet_proper2/35dd0jma",
    #     "jjshoots/pybullet_proper2/3rhmkt0r",
    #     "jjshoots/pybullet_proper2/2z866q69",
    #     "jjshoots/pybullet_proper2/1kvqykmn",
    #     "jjshoots/pybullet_proper2/36uah4ty",
    #     "jjshoots/pybullet_proper2/1sz3rlfh",
    #     "jjshoots/pybullet_proper2/10whr5h3",
    # ]
    # run_list.append(runs)
    # baselines = {}
    # baselines["Oracle 1"] = 1508.0
    # baselines["Oracle 2"] = 2246.0
    # baseline_list.append(baselines)
    # env_list.append("HopperPyBulletEnv-v0")

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
    runs["CCGE no bias w/ Oracle 2"] = [
        "jjshoots/pybullet_proper2/141jvkir",
        "jjshoots/pybullet_proper2/35iqj6jq",
        "jjshoots/pybullet_proper2/1yewmolf",
        "jjshoots/pybullet_proper2/2g5pewnq",
        "jjshoots/pybullet_proper2/1urylorh",
        "jjshoots/pybullet_proper2/onk3hzdp",
        "jjshoots/pybullet_proper2/3e102eof",
        "jjshoots/pybullet_proper2/369n9lgm",
        "jjshoots/pybullet_proper2/s34cof1j",
        "jjshoots/pybullet_proper2/2wszexx3",
    ]
    runs["CCGE bias w/ Oracle 2, good"] = [
        "jjshoots/pybullet_proper2/3gp5nyen",
        "jjshoots/pybullet_proper2/2ractkua",
        "jjshoots/pybullet_proper2/3hmum2n0",
        "jjshoots/pybullet_proper2/21e9dne8",
        "jjshoots/pybullet_proper2/7tvp5l5m",
        "jjshoots/pybullet_proper2/2qr7wlle",
        "jjshoots/pybullet_proper2/y340wa16",
        "jjshoots/pybullet_proper2/1j1ltnxw",
        "jjshoots/pybullet_proper2/2mlpp813",
        "jjshoots/pybullet_proper2/wddv49zf",
    ]
    runs["CCGE bias w/ Oracle 2, bad"] = [
        "jjshoots/pybullet_proper2/33rckw0u",
        "jjshoots/pybullet_proper2/3tvk325r",
        "jjshoots/pybullet_proper2/5yjlolf6",
        "jjshoots/pybullet_proper2/1ixju3q6",
        "jjshoots/pybullet_proper2/1kud7yko",
        "jjshoots/pybullet_proper2/32ikucco",
        "jjshoots/pybullet_proper2/18bm2agp",
        "jjshoots/pybullet_proper2/3o2l899x",
        "jjshoots/pybullet_proper2/24d0q0as",
    ]
    runs["CCGE no bias w/ Oracle 1"] = [
        "jjshoots/pybullet_proper2/39k9e4ha",
        "jjshoots/pybullet_proper2/3n84n1m1",
        "jjshoots/pybullet_proper2/24u31w63",
        "jjshoots/pybullet_proper2/3bqqcm3w",
        "jjshoots/pybullet_proper2/1nclenww",
        "jjshoots/pybullet_proper2/3yh1ie5b",
        "jjshoots/pybullet_proper2/1qq0lp04",
        "jjshoots/pybullet_proper2/16eo29of",
        "jjshoots/pybullet_proper2/3u3ig6md",
        "jjshoots/pybullet_proper2/3333407d",
    ]
    run_list.append(runs)
    baselines = {}
    baselines["Oracle 1"] = 796.0
    baselines["Oracle 2"] = 787.0
    baseline_list.append(baselines)
    env_list.append("HalfCheetahPyBulletEnv-v0")

    # # Walker2D
    # runs = {}
    # runs["SAC"] = [
    #     "jjshoots/pybullet_proper2/iv83lvpe",
    #     "jjshoots/pybullet_proper2/1nugaeda",
    #     "jjshoots/pybullet_proper2/1l99vuyl",
    #     "jjshoots/pybullet_proper2/lgab8igk",
    #     "jjshoots/pybullet_proper2/xwziupc5",
    #     "jjshoots/pybullet_proper2/3d59105o",
    #     "jjshoots/pybullet_proper2/13p6apja",
    #     "jjshoots/pybullet_proper2/3bb8whna",
    #     "jjshoots/pybullet_proper2/1gsqjuow",
    #     "jjshoots/pybullet_proper2/1vyczwt2",
    #     "jjshoots/pybullet_proper2/1nd3v4dh",
    #     "jjshoots/pybullet_proper2/2000jz2c",
    # ]
    # runs["CCGE no bias w/ Oracle 2"] = [
    #     "jjshoots/pybullet_proper2/2zwslh8j",
    #     "jjshoots/pybullet_proper2/287kdug2",
    #     "jjshoots/pybullet_proper2/257wwxpv",
    #     "jjshoots/pybullet_proper2/3gukol9m",
    #     "jjshoots/pybullet_proper2/38ga2m4a",
    #     "jjshoots/pybullet_proper2/12p9irs1",
    #     "jjshoots/pybullet_proper2/3uvrbauh",
    #     "jjshoots/pybullet_proper2/3mgsmprg",
    #     "jjshoots/pybullet_proper2/5kkyxr1a",
    #     "jjshoots/pybullet_proper2/8ob04seg",
    # ]
    # runs["CCGE bias w/ Oracle 2"] = [
    #     "jjshoots/pybullet_proper2/136to5y6",
    #     "jjshoots/pybullet_proper2/22vfld1r",
    #     "jjshoots/pybullet_proper2/2ef2lzva",
    #     "jjshoots/pybullet_proper2/3t9ge95c",
    #     "jjshoots/pybullet_proper2/2hgrh2z1",
    #     "jjshoots/pybullet_proper2/1pjvxet6",
    #     "jjshoots/pybullet_proper2/3mv43nah",
    #     "jjshoots/pybullet_proper2/7qfnbjot",
    #     "jjshoots/pybullet_proper2/3f3su60g",
    #     "jjshoots/pybullet_proper2/3tme0soy",
    # ]
    # runs["CCGE no bias w/ Oracle 1"] = [
    #     "jjshoots/pybullet_proper2/14g2rr00",
    #     "jjshoots/pybullet_proper2/2mkj2pue",
    #     "jjshoots/pybullet_proper2/1f445wsa",
    #     "jjshoots/pybullet_proper2/fenlrr1h",
    #     "jjshoots/pybullet_proper2/v1lnczb9",
    #     "jjshoots/pybullet_proper2/1wvndz2h",
    #     "jjshoots/pybullet_proper2/c4y2o7or",
    #     "jjshoots/pybullet_proper2/j7m6vphn",
    #     "jjshoots/pybullet_proper2/35cmivtx",
    #     "jjshoots/pybullet_proper2/3dco4ecb",
    # ]
    # run_list.append(runs)
    # baselines = {}
    # baselines["Oracle 1"] = 1506.0
    # baselines["Oracle 2"] = 1536.0
    # baseline_list.append(baselines)
    # env_list.append("Walker2DPyBulletEnv-v0")

    for runs, env_name, baselines in zip(run_list, env_list, baseline_list):
        compute_plots(runs, env_name, baselines)
