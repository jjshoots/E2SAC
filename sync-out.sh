#!/bin/sh
rsync -av --exclude-from='rsync_ignore_out.txt' ./ availab-dl1:~/Sandboxes/e2SAC/ --delete
ssh availab-dl1 'cd ~/Sandboxes/e2SAC/ && source ./venv/bin/activate.fish && echo y | wandb sync --clean'
rsync -av --exclude-from='rsync_ignore_out.txt' ./ availab-dl2:~/Sandboxes/e2SAC/ --delete
ssh availab-dl2 'cd ~/Sandboxes/e2SAC/ && source ./venv/bin/activate.fish && echo y | wandb sync --clean'
