#!/bin/sh
# rsync -rav --progress --include-from='rsync_include_in.txt' --exclude='*' availab-dl1:~/Sandboxes/e2SAC/ ./
# rsync -rav --progress --include-from='rsync_include_in.txt' --exclude='*' availab-dl2:~/Sandboxes/e2SAC/ ./
rsync -avr --progress sulis:~/e2SAC/weights/Version673181 ./weights/Version673181
rsync -avr --progress sulis:~/e2SAC/optim_weights/Version673181 ./optim_weights/Version673181
