#!/bin/sh
rsync -av --progress --stats --include-from='rsync_include_in.txt' availab-dl1:~/Sandboxes/e2SAC/ ./
rsync -av --progress --stats --include-from='rsync_include_in.txt' availab-dl2:~/Sandboxes/e2SAC/ ./
