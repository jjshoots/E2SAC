#!/bin/sh
rsync -avr --progress --stats --exclude-from='rsync_ignore_out.txt' ./ availab-dl1:~/Sandboxes/e2SAC/ --delete
# rsync -avr --progress --stats --exclude-from='rsync_ignore_out.txt' ./ availab-dl2:~/Sandboxes/e2SAC/ --delete
# rsync -avr --progress --stats --exclude-from='rsync_ignore_out.txt' ./ availab-dl3:~/Sandboxes/e2SAC/ --delete
# rsync -avr --progress --stats --exclude-from='rsync_ignore_out.txt' ./ availab-dl4:~/Sandboxes/e2SAC/ --delete
