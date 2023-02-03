#!/bin/sh
rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/weights/Version593596/ ./weights/Version593596
# rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl3:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl4:~/Sandboxes/e2SAC/weights/ ./weights

wingman-compress-weights
