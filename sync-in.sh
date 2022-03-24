#!/bin/sh
rsync -av --progress --include-from='rsync_include_in.txt' --exclude='*' availab-dl1:~/Sandboxes/e2SAC/ ./
rsync -av --progress --include-from='rsync_include_in.txt' --exclude='*' availab-dl2:~/Sandboxes/e2SAC/ ./
