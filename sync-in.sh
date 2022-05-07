#!/bin/sh
# rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/optim_weights/ ./optim_weights
# rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/optim_weights/ ./optim_weights
rsync -avr --progress taijunjet1997@35.208.249.167:~/e2SAC/weights/ ./weights
rsync -avr --progress taijunjet1997@35.208.249.167:~/e2SAC/optim_weights/ ./optim_weights

