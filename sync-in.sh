#!/bin/sh
rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/suboptimal_policies/ ./suboptimal_policies
# rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/weights/ ./weights
# rsync -avr --progress availab-dl1:~/Sandboxes/e2SAC/optim_weights/ ./optim_weights
# rsync -avr --progress availab-dl2:~/Sandboxes/e2SAC/optim_weights/ ./optim_weights
