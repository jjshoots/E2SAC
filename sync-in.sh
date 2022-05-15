#!/bin/sh
# rsync -rav --progress --include-from='rsync_include_in.txt' --exclude='*' availab-dl1:~/Sandboxes/e2SAC/ ./
# rsync -rav --progress --include-from='rsync_include_in.txt' --exclude='*' availab-dl2:~/Sandboxes/e2SAC/ ./
rsync -rav --progress --exclude='*' --include-from='rsync_include_in.txt' sulis:~/e2SAC/ ./
