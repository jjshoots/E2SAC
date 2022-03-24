#!/usr/bin/env python3
import os
import time

import numpy as np
import torch

class Helpers:
    def __init__(
        self,
        mark_number,
        version_number,
        weights_location,
        epoch_interval=-1,
        batch_interval=-1,
        max_skips=5,
        greater_than=0.0,
        increment=True,
    ):
        """
        Helper class for handling loss recording and weights file management
        """
        assert (
            epoch_interval * batch_interval < 0
        ), "epoch_interval or batch_interval must be positive number"

        self.iter_passed = 0
        self.running_loss = 0
        self.greater_than = greater_than
        self.lowest_running_loss = 0

        self.increment = increment
        self.interval = epoch_interval if epoch_interval > 0 else batch_interval
        self.use_epoch = epoch_interval > 0
        self.previous_save_step = 0
        self.skips = 0
        self.max_skips = max_skips

        # weight file variables
        self.directory = os.path.dirname(__file__)
        self.weights_location = os.path.join(self.directory, f"../{weights_location}")
        self.mark_number = mark_number
        self.version_number = version_number
        self.weights_file = os.path.join(
            self.weights_location,
            f"Version{self.version_number}/weights{self.mark_number}.pth",
        )
        self.weights_file_short = weights_location

        # record that we're in a new training session
        if not os.path.isfile(
            os.path.join(
                self.weights_location, f"Version{self.version_number}/training_log.txt"
            )
        ):
            print(
                f"No weights directory for {self.weights_file_short}/Version{self.version_number}, generating new one in 3 seconds."
            )
            time.sleep(3)
            os.makedirs(
                os.path.join(self.weights_location, f"Version{self.version_number}")
            )

        f = open(
            os.path.join(
                self.weights_location, f"Version{self.version_number}/training_log.txt"
            ),
            "a",
        )
        f.write(f"New Session, Net Version {self.version_number} \n")
        f.write(f"Epoch, Batch, Running Loss, Lowest Running Loss, Mark Number \n")
        f.close()

    def training_checkpoint(self, loss, batch, epoch, readout=True):
        """
        call inside training loop
        helps to display training progress and save any improvement
        returns a string path or -1 depending if we need to save or not
        """

        readout = self.increment and readout
        step = epoch if self.use_epoch else batch

        if step % self.interval == 0:
            self.running_loss = loss
            self.iter_passed = 1.0
        else:
            self.running_loss += loss
            self.iter_passed += 1.0

        if step % self.interval == 0 and step != 0 and step != self.previous_save_step:
            self.previous_save_step = step

            # at the moment, no way to evaluate the current state of training, so we just record the current running loss
            avg_loss = self.running_loss / self.iter_passed
            self.lowest_running_loss = (
                avg_loss
                if (self.lowest_running_loss == 0.0)
                else self.lowest_running_loss
            )

            if readout:
                # print status
                print(
                    f"Epoch {epoch}; Batch Number {batch}; Running Loss {avg_loss:.5f}; Lowest Running Loss {self.lowest_running_loss:.5f}"
                )
                # record training log
                f = open(
                    os.path.join(
                        self.weights_location,
                        f"Version{self.version_number}/training_log.txt",
                    ),
                    "a",
                )
                f.write(
                    f"{epoch}, {batch}, {self.running_loss}, {self.lowest_running_loss}, {self.mark_number} \n"
                )
                f.close()

            # save the network if the running loss is lower than the one we have
            if avg_loss < self.lowest_running_loss:
                self.skips = 0

                self.lowest_running_loss = avg_loss

                self.mark_number += 1 if self.increment else 0

                # regenerate the weights_file path
                self.weights_file = os.path.join(
                    self.weights_location,
                    f"Version{self.version_number}/weights{self.mark_number}.pth",
                )
                if readout:
                    print(
                        f"New lowest point, saving weights to: {self.weights_file_short}/weights{self.mark_number}.pth"
                    )
                return self.weights_file
            else:
                self.skips += 1

            # save the network to intermediary if we crossed the max number of skips
            if self.skips >= self.max_skips:
                self.skips = 0

                self.weights_file = os.path.join(
                    self.weights_location,
                    f"Version{self.version_number}/weights_intermediary.pth",
                )
                if readout:
                    print(
                        f"Passed {self.max_skips} intervals without saving so far, saving weights to: /weights_intermediary.pth"
                    )
                return self.weights_file

        # return -1 if we are not returning the weights file
        return -1

    def write_auxiliary(
        self, data: np.ndarray, variable_name: str, precision: str = "%1.3f"
    ) -> None:
        assert len(data.shape) == 1, "Data must be only 1 dimensional ndarray"
        filename = os.path.join(
            self.weights_location, f"Version{self.version_number}/{variable_name}.csv"
        )
        with open(filename, "ab") as f:
            np.savetxt(f, [data], delimiter=",", fmt=precision)

    def get_weight_file(self, latest=True):
        """
        retrieves the latest weight file based on mark and version number
        weight location is location where all weights of all versions are stored
        version number for new networks, mark number for training
        """
        # if we are not incrementing and the file doesn't exist, just exit
        if not self.increment:
            if not os.path.isfile(self.weights_file):
                return -1

        # if we don't need the latest file, get the one specified
        if not latest:
            if os.path.isfile(self.weights_file):
                self.weights_file = os.path.join(
                    self.weights_location,
                    f"Version{self.version_number}/weights{self.mark_number}.pth",
                )
                return self.weights_file

        # while the file exists, try to look for a file one version later
        while os.path.isfile(self.weights_file):
            self.mark_number += 1
            self.weights_file = os.path.join(
                self.weights_location,
                f"Version{self.version_number}/weights{self.mark_number}.pth",
            )

        # once the file version doesn't exist, decrement by one and use that file
        self.mark_number -= 1
        self.weights_file = os.path.join(
            self.weights_location,
            f"Version{self.version_number}/weights{self.mark_number}.pth",
        )

        # if there's no files, ignore, otherwise, print the file
        if os.path.isfile(self.weights_file):
            print(
                f"Using weights file: /{self.weights_file_short}/Version{self.version_number}/weights{self.mark_number}.pth"
            )
            return self.weights_file
        else:
            print(f"No weights file found, generating new one during training.")
            return -1


#################################################################################################
#################################################################################################
# STATIC FUNCTIONS
#################################################################################################
#################################################################################################


def get_device():
    """
    gets gpu if available otherwise cpu
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    print("-----------------------")
    print("Using Device", device)
    print("-----------------------")

    return device


def gpuize(input, device):
    if torch.is_tensor(input):
        if input.device == device:
            return input.float()
        return input.to(device).float()
    return torch.tensor(input).float().to(device)


def cpuize(input):
    if torch.is_tensor(input):
        return input.detach().cpu().numpy()
    else:
        return input


def network_stats(network, input_image):
    from pthflops import count_ops

    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    count_ops(network, input_image)
    print(f"Total number of Parameters: {total_params}")
