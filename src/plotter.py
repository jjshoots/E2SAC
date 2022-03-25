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


def get_log(run_uri, keys):
    assert isinstance(keys, list), "keys must be a list."
    api = wandb.Api()
    run = api.run(run_uri)
    history = run.scan_history(keys=keys)

    data = {}
    for key in keys:
        array = np.array([row[key] for row in history])
        data[key] = array

    return data


if __name__ == "__main__":
    # parameters
    num_episodes = 501

    # list of algorithms and their corresponding uris
    runs = {}
    runs["E2SAC"] = [
    ]
    runs["SAC"] = [
        # ANT
        "jjshoots/e2SAC_pybullet/27cjzt50",
        "jjshoots/e2SAC_pybullet/6vgj57fp",
        "jjshoots/e2SAC_pybullet/1l2a1jz3",
        "jjshoots/e2SAC_pybullet/110jrok2",
        "jjshoots/e2SAC_pybullet/1xyiilpo",
        "jjshoots/e2SAC_pybullet/2x5hkmbc",
        "jjshoots/e2SAC_pybullet/13r7w035",
        "jjshoots/e2SAC_pybullet/3pmbwojt",
    ]

    # list of algorithms we have
    algorithms = [key for key in runs]

    # load scores as dictionary mapping algorithms to their scores
    # each score is of size (num_runs x num_games x num_recordings)
    scores = {}
    for algorithm in runs:
        score = []
        for run_uri in runs[algorithm]:
            score.append(get_log(run_uri, ["eval_perf"])["eval_perf"][:num_episodes])

        # stack along num_runs axis
        score = np.stack(score, axis=0)
        # expand along num_games axis
        score = np.expand_dims(score, axis=1)

        # add to overall scores
        scores[algorithm] = score

    episode = np.arange(0, num_episodes, 20)
    ale_frames_scores_dict = {
        algorithm: score[:, :, episode] for algorithm, score in scores.items()
    }
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])]
    )
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        ale_frames_scores_dict, iqm, reps=50000
    )

    # plot sample efficiency curve
    plot_utils.plot_sample_efficiency_curve(
        episode + 1,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel=r"Number of episodes",
        ylabel="Interquartile Mean (IQM)",
    )

    # form the legend
    color_dict = dict(zip(algorithms, sns.color_palette("colorblind")))
    fake_patches = [
        patches.Patch(color=color_dict[alg], alpha=0.75) for alg in algorithms
    ]
    legend = plt.legend(
        fake_patches,
        algorithms,
        loc="upper center",
        fancybox=True,
        ncol=len(algorithms),
        fontsize="x-large",
        bbox_to_anchor=(0.5, 1.1),
    )

    plt.show()
