#!/bin/sh
# rsync -avr --exclude-from='rsync_ignore_out.txt' ./ availab-dl1:~/Sandboxes/e2SAC/ --delete
# rsync -avr --exclude-from='rsync_ignore_out.txt' ./ availab-dl2:~/Sandboxes/e2SAC/ --delete
# rsync -avr --exclude-from='rsync_ignore_out.txt' ./ sulis:~/e2SAC/ --delete
rsync -avr --exclude-from='rsync_ignore_out.txt' ./ zeus:~/e2SAC/ --delete
# rsync -avr --exclude-from='rsync_ignore_out.txt' ./ taijunjet1997@35.208.249.167:~/e2SAC/ --delete
