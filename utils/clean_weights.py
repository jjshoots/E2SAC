import glob
import os
import shutil

# all versions
dirs = glob.glob("./weights/Version*")

# which folders to delete
to_delete = []

# start going through each folder
for dir in dirs:
    weights = glob.glob(os.path.join(dir, "weights*.pth"))

    # if there's less than 2 files, ie: weights0.pth only
    if len(weights) < 2:
        to_delete.append(dir)
        continue

    # start finding the latest file
    num = 0
    has_num = True
    latest = ""
    while has_num:
        latest = os.path.join(dir, f"weights{num}.pth")

        # find the latest version and remove it from the list
        if latest in weights:
            num += 1
            has_num = True
        else:
            num -= 1
            has_num = False
            latest = os.path.join(dir, f"weights{num}.pth")
            weights.remove(latest)

    # remove the intermediary from our list if it's there
    intermediary = os.path.join(dir, f"weights_intermediary.pth")
    if intermediary in weights:
        weights.remove(intermediary)

    # for the remainder of the list, delete all the weights
    for weight in weights:
        os.remove(weight)

    # rename the latest weights to 0
    os.rename(latest, os.path.join(dir, "weights0.pth"))

# delete optim weights and weights together
for dir in to_delete:
    version = os.path.basename(dir)
    shutil.rmtree(dir, ignore_errors=False, onerror=None)
    shutil.rmtree(f"./optim_weights/{version}", ignore_errors=False, onerror=None)
