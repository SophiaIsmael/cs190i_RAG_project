"""
This module contains the structure of a BASEModel that will be implemented in the YOLO model
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, product
from pathlib import Path
from typing import Any

import matplotlib.axis as Axis
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.utils as tvutils
from PIL import Image
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# from tqdm import tqdm
from tqdm.notebook import tqdm


class BASEModel(nn.Module):
    """TODO"""

    def __init__(
        self,
        multistep_lr_schedule: list[int] = None,
        multistep_lr_gamma: float = 0.1,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.2,
        device="cpu",
    ):
        """
        calls super __init__ and sets up parameters

        Args:
            multistep_lr_schedule (list[int]): multistep learning rate default set to None
            learning_rate (float): learning rate default set to 1e-3,
            dropout_rate (float): dropout rate default set to 0.2
            device (str): device default set to cpu

        """
        # calls the __init__ of the super class (torch.nn) to initialize the class
        super().__init__()

        # initialize the learning rate, dropout rate, device (cpu or gpu), multistep learning rate from object instantiation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.device = device
        self.multistep_lr_schedule = multistep_lr_schedule
        self.multistep_lr_gamma = multistep_lr_gamma

    def configure_optimizer(self):
        """
        - sets up optimizer
        - sets up multistep learning rate (if there is one, set to None if not)
        """
        # sets optimizer to Adam
        self.optimizer = torch.optim.Adam(
            self.get_parameter_list(),
            lr=self.learning_rate,
        )
        if self.multistep_lr_schedule:
            # if multistep learning rate schedule was initialized, set the learning rate scheduler, ow: set to none
            self.lr_scheduler = MultiStepLR(
                self.optimizer,
                self.multistep_lr_schedule,
                gamma=self.multistep_lr_gamma,
            )
        else:
            self.lr_scheduler = None

    def get_optimizer_state_dict(self):
        """getter function to return state of optimizer --> used for checkpointing"""
        return self.optimizer.state_dict()

    def load_optimizer_state_dict(self, optimizer_state_dict: dict[str, Any]):
        """load the state of optimizer --> used for checkpointing, possible when continuing from a pause"""
        self.optimizer.load_state_dict(optimizer_state_dict)

    def training_step(
        self,
        training_batch: torch.tensor,
        target_batch: Any,
    ) -> torch.Tensor:
        """

        - propagates forward step on training_batch
        - back propagation step
        - calcs loss
        - updates learning rate scheduler

        Args:
            training_batch (torch.tensor): tensor of image tensors
            target_batch (list[ list [dict[str, Any]]])

        Returns:
            torch.Tensor: returns the loss

        """

        # clear gradients from previous step
        self.optimizer.zero_grad()

        # apply the model on training batch
        output_batch = self.forward(training_batch)

        # after applying forward, we should calc loss using the meta_data and return the loss
        loss = self.loss_fn(
            training_batch,
            output_batch,
            target_batch,
        )

        ### Backpropagation ###
        # sum the loss of the whole batch
        loss = torch.sum(loss)

        # backpropagation pass to compute the gradient loss
        loss.backward()

        # uses loss gradient to update the weights of the model
        self.optimizer.step()

        ####

        if self.lr_scheduler:
            # update learning rate scheduler if it exists
            self.lr_scheduler.step()

        return loss

    def evaluation_step(self, eval_batch: torch.tensor):
        """TODO"""

    def test_step(self, eval_batch: torch.tensor):
        """TODO"""

        
    # ###########################################
    # abstract methods
    # ###########################################
    def get_hyper_parameters(self):
        """TODO"""
        raise NotImplementedError("abstract method must be implemented by child class!")

    def get_parameter_list(self):
        """TODO"""
        raise NotImplementedError("abstract method must be implemented by child class!")

    def get_state_dict(self):
        """TODO"""
        raise NotImplementedError("abstract method must be implemented by child class!")

    def loss_fn(
        self,
        training_batch,
        output_batch,
        target_batch,
    ):
        """TODO"""
        raise NotImplementedError("abstract method must be implemented by child class!")
