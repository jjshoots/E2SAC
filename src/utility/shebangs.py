import os
import argparse

import yaml
import wandb

from ai_lib.helpers import *

def shutdown_handler(*_):
    print("ctrl-c invoked")
    exit(0)


def arg_parser():
    parser = argparse.ArgumentParser(description='IDK yet.')

    parser.add_argument("--wandb", type=bool, nargs='?',
                            const=True, default=False,
                            help="Log to W and B.")

    parser.add_argument("--debug", type=bool, nargs='?',
                            const=True, default=False,
                            help="Activate debug mode.")

    parser.add_argument("--display", type=bool, nargs='?',
                            const=True, default=False,
                            help="Set to display mode.")

    parser.add_argument("--train", type=bool, nargs='?',
                            const=True, default=False,
                            help="Train the network.")

    parser.add_argument("--shutdown", type=bool, nargs='?',
                            const=True, default=False,
                            help="Shutdown after training.")

    parser.add_argument("--name", type=str, nargs='?',
                            const=False, default='',
                            help="Name of the run in wandb.")

    parser.add_argument("--id", type=str, nargs='?',
                            const=False, default='',
                            help="ID of the run in wandb.")

    return parser.parse_args()


def parse_set_args():
    # parse arguments
    args = arg_parser()

    # parse settings
    set = None
    with open(os.path.join(os.path.dirname(__file__), '../settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        settings['debug'] = True if args.debug else False
        settings['wandb'] = True if args.wandb else False
        settings['num_envs'] = 1 if args.display else settings['num_envs']
        settings['device'] = get_device()
        settings['iter_per_epoch'] = int(settings['buffer_size'] / settings['batch_size'])
        settings['step_sched_num'] = settings['iter_per_epoch'] * settings['epochs'] / settings['scheduler_steps']
        settings['buffer_size'] = settings['buffer_size_debug'] if args.debug else settings['buffer_size']
        settings['transitions_per_epoch'] = settings['buffer_size_debug'] if args.debug else settings['transitions_per_epoch']
        settings['num_actions'] = 1
        set = argparse.Namespace(**settings)

    # weights and biases
    if set.wandb and args.train:
        wandb.init(
            project=set.wandb_project,
            entity='jjshoots',
            config=settings,
            name=args.name + ', v=' + set.net_version if args.name != '' else None,
            id=args.id if args.id != '' else None)

    return set, args
