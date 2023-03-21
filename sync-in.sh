#!/bin/sh
rsync -avr --progress prophet:~/e2SAC/weights/Version259518/ ./weights/Version259518
# rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/weights/Version259158/ ./weights/Version259158
# rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl3:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl4:~/Sandboxes/e2SAC/weights/ ./weights

wingman-compress-weights
