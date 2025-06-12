"""TODO"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import chain, product
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.utils as tvutils
from base_model import BASEModel
from PIL import Image
from pycocotools.coco import COCO
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# from tqdm import tqdm
from tqdm import tqdm


class Trainer:
    """TODO"""

    def __init__(
        self,
        /,
        *,
        model: BASEModel,
        train_dataloader: DataLoader,
        num_training_batches: int,
        checkpoint_dir: str,
        keep_only_last_checkpoints: int = 0,
        save_checkpoint_every_epochs: int = -1,  # -1 means save only the last checkpoint
        log_dir: str = None,
    ):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.keep_only_last_checkpoints = keep_only_last_checkpoints
        self.train_dataloader = train_dataloader
        self.num_training_batches = num_training_batches
        self.save_checkpoint_every_epochs = save_checkpoint_every_epochs
        self.log_dir = log_dir

    def get_model_summary(self) -> tuple[int, dict[str, int]]:
        """TODO"""
        total_param_count = sum(p.numel() for p in self.model.parameters())

        params_summary = {
            p_name: p_val.numel() for p_name, p_val in self.model.state_dict().items()
        }

        print(f"total parameter count: {total_param_count:,}")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"\t\tname: {name}, number of parameters: {param.data.numel():,}")

        return total_param_count, params_summary

    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
    ) -> None:

        if not self.checkpoint_dir:
            return

        # ensure checkpoint dir exists
        checkpoint_dir_path = Path(self.checkpoint_dir)
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_dir_path / Path(
            type(self.model).__name__ + "." + str(datetime.now().timestamp()) + ".ckpt"
        )

        # write the checkpoint to the file
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.get_state_dict(),
                "optimizer_state_dict": self.model.get_optimizer_state_dict(),
                "loss": loss,
                "hyper_params": self.model.get_hyper_parameters(),
            },
            str(checkpoint_file),
        )

        existing_checkpoints = sorted(
            checkpoint_dir_path.glob(type(self.model).__name__ + ".*.ckpt"),
            key=lambda x: os.path.getmtime(x),
        )

        #  delete all but most recent keep_only_last_checkpoints
        if self.keep_only_last_checkpoints > 0:
            num_ckpt_to_delete = (
                len(existing_checkpoints) - self.keep_only_last_checkpoints
            )
            if num_ckpt_to_delete > 0:
                for old_ckpt in existing_checkpoints[0:num_ckpt_to_delete]:
                    # print(f"deleteing: {old_ckpt}")
                    old_ckpt.unlink(missing_ok=True)

    @classmethod
    def load_from_checkpoint(
        cls,
        model_module: str,
        model_class_name: str,
        checkpoint_file: str,
        device: torch.device,
        learning_rate: float = None,
        multistep_lr_schedule: list[int] = None,
        multistep_lr_gamma: float = None,
    ) -> tuple[type[BASEModel], float, float]:
        """TODO"""
        checkpoint_path = Path(checkpoint_file)
        assert checkpoint_path.exists(), f"File not found: {checkpoint_path}"

        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=device
        )

        print(f"{checkpoint['hyper_params']=}\n\n")

        module = importlib.import_module(model_module)
        model_class = getattr(module, model_class_name)

        hyper_params = checkpoint["hyper_params"]

        if multistep_lr_schedule:
            hyper_params["multistep_lr_schedule"] = multistep_lr_schedule
        if multistep_lr_gamma:
            hyper_params["multistep_lr_gamma"] = multistep_lr_gamma
        if learning_rate:
            hyper_params["learning_rate"] = learning_rate

        model = model_class(**hyper_params, device=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.load_optimizer_state_dict(checkpoint["optimizer_state_dict"])
        model.train()

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        return model, epoch, loss

    @classmethod
    def resume_train_cycle(
        cls,
        /,
        *,
        model_module: str,
        model_class_name: str,
        train_dataloader: DataLoader,
        num_training_batches: int,
        epochs,
        checkpoint_file: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        multistep_lr_schedule: list[int] = None,
        multistep_lr_gamma: float = None,
        checkpoint_dir: str = None,
        save_checkpoint_every_epochs: int = -1,  # -1 means save only the last checkpoint
        keep_only_last_checkpoints: int = 0,
        log_dir: str = None,
    ) -> None:
        """TODO"""
        model, last_epoch, loss = cls.load_from_checkpoint(
            model_module,
            model_class_name,
            checkpoint_file,
            device,
            learning_rate,
            multistep_lr_schedule,
            multistep_lr_gamma,
        )

        print(f"{last_epoch=}")
        trainer = cls(
            model=model,
            train_dataloader=train_dataloader,
            num_training_batches=num_training_batches,
            checkpoint_dir=checkpoint_dir,
            keep_only_last_checkpoints=keep_only_last_checkpoints,
            save_checkpoint_every_epochs=save_checkpoint_every_epochs,
            log_dir=log_dir,
        )

        trainer.perform_train_cycle(epochs, start_epoch_index=last_epoch + 1)

    def perform_train_cycle(
        self,
        epochs: int,
        start_epoch_index: int = 0,
    ) -> None:
        """TODO"""

        self.get_model_summary()

        writer = None
        if self.log_dir:
            log_subdir_path = Path(self.log_dir) / Path(
                f"{type(self.model).__name__}.{int(datetime.now().timestamp()):x}"
            )
            #  make sure the log dir exists
            log_subdir_path.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(log_subdir_path), flush_secs=30)

        # add hyper params to logs
        # if writer:
        #     writer.add_hparams(model.get_hyper_parameters(), {})

        loss = torch.zeros((1))
        tqdm_epoch = tqdm(
            range(start_epoch_index, start_epoch_index + epochs),
            total=epochs,
        )
        last_epoch = 0
        for epoch in tqdm_epoch:
            # switch model mode to training:
            self.model.train()
            last_epoch = epoch

            processed_batches = 0
            epoch_loss = 0
            train_data_loader_iter = iter(self.train_dataloader)
            with tqdm(
                train_data_loader_iter,
                leave=False,
                desc=f"{epoch=}",
                total=self.num_training_batches,
            ) as tqdm_batch_loader:
                for training_batch, target_batch in tqdm_batch_loader:
                    # move the image tensor to model.device (e.g. mps)
                    # train_tensor = train_tensor.to(device=model.device)

                    processed_batches += 1
                    if processed_batches > self.num_training_batches:
                        break

                    # after applying forward, we should calc loss using the meta_data and return the loss
                    loss = self.model.training_step(
                        training_batch,
                        target_batch,
                    )

                    # # Backpropagation
                    loss = torch.sum(loss)
                    epoch_loss += loss

                    if writer:
                        writer.add_scalar(
                            "train/batch_loss",
                            loss,
                            epoch * self.num_training_batches + processed_batches,
                        )

                # tqdm_batch_loader.container.close()
                epoch_loss /= self.num_training_batches
                if writer:
                    writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

            if (
                self.save_checkpoint_every_epochs > 0
                and epoch > 0
                and epoch % self.save_checkpoint_every_epochs == 0
            ):
                self._save_checkpoint(
                    epoch=last_epoch,
                    loss=loss,
                )
            if writer:
                writer.flush()

        # save final checkpoint
        self._save_checkpoint(
            epoch=last_epoch,
            loss=loss,
        )
