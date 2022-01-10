#!/bin/sh
rsync -av --exclude-from='rsync_ignore_in.txt' availab-dl1:~/Sandboxes/e2SAC/ ./
rsync -av --exclude-from='rsync_ignore_in.txt' availab-dl2:~/Sandboxes/e2SAC/ ./
