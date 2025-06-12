from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from itertools import chain, product
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from base_model import BASEModel
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

# from tqdm import tqdm
from tqdm.notebook import tqdm


class RAGAutoencoder(BASEModel):
    """

    Args:
        token_embedding_dim (int): dimensions of the token embeddings matrix
        max_context_length (int): maximum length of the context
        encoder_embeddings_size (int): size of the embedding space
        multistep_lr_schedule (list[int]): multistep learning rate scheduler
        multistep_lr_gamma (float)
        learning_rate (float): learning rate of the model
        dropout_rate (float): dropout rate of the model
        device (str): device used by the model
    """

    def __init__(
        self,
        /,
        *,
        token_embedding_dim: int,
        max_context_length: int,
        encoder_embeddings_size: int = 1024,
        multistep_lr_schedule: list[int] = None,
        multistep_lr_gamma: float = 0.1,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.25,
        device="cpu",
    ):
        super().__init__(
            multistep_lr_schedule=multistep_lr_schedule,
            multistep_lr_gamma=multistep_lr_gamma,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.max_context_length = max_context_length
        self.encoder_embeddings_size = encoder_embeddings_size

        self.token_embedding_dim = token_embedding_dim

        # previous input and embeddings tensors to be excluded from parameters
        # used only during training
        # self.previous_input = None
        self.register_buffer("previous_input_tensor", None, persistent=False)
        self.register_buffer("previous_input_id", None, persistent=False)
        # self.previous_encoder_output = None
        self.register_buffer("previous_encoder_output", None, persistent=False)

        self.input_rms_norm = nn.RMSNorm(
            # [self.max_context_length, self.token_embedding_dim]
            [self.token_embedding_dim]
        )

        # the model will reduce the token embeddings dim from 2048 to 64 latent embedding space
        self.token_embeddings_latent_dim = 64

        # compression across token dimm, linear NN
        self.token_embedding_compressor = nn.Linear(
            self.token_embedding_dim,
            self.token_embeddings_latent_dim,
        )

        # compression over context length
        # this is the output embeddings
        self.context_len_compressor = nn.Linear(
            self.max_context_length * self.token_embeddings_latent_dim,
            self.encoder_embeddings_size,
        )

        # context decompressor
        self.context_len_decompressor = nn.Linear(
            self.encoder_embeddings_size,
            self.max_context_length * self.token_embeddings_latent_dim,
        )

        # decompressor to token dimm, linear NN
        self.token_embedding_decompressor = nn.Linear(
            self.token_embeddings_latent_dim,
            self.token_embedding_dim,
        )

        #
        #
        #
        # dropout layer
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # finally call the base_model configure_optimizer
        self.configure_optimizer()

    def get_parameter_list(self):
        """
        - gets a list of models params

        Returns:
        list: returns a list containing the model (conv blocks, maxpool layers), the first connected layer,
            the dropout layer, and the bbox/class predictions
        """
        return list(
            chain(
                self.input_rms_norm.parameters(),
                self.token_embedding_compressor.parameters(),
                self.context_len_compressor.parameters(),
                self.context_len_decompressor.parameters(),
                self.token_embedding_decompressor.parameters(),
                self.dropout.parameters(),
            ),
        )

    def get_state_dict(self) -> dict[str, Any]:
        """TODO"""
        return self.state_dict()

    def forward(self, batch_input_tensor) -> tuple[torch.Tensor]:
        """TODO"""
        # normalize inputs using RMSNorm to stabilize gradients and
        # speed up learning
        _input_id, _inputs = batch_input_tensor
        
        _norm_inputs = self.input_rms_norm(_inputs)

        # print(f"{_inputs.size()=}")
        _output = self.token_embedding_compressor(_norm_inputs)

        num_batches, ctx_len, emb = _output.size()
        _output = _output.view(num_batches, ctx_len * emb).contiguous()
        # adding some nonlinearity
        _output = nn.functional.silu(_output) + _output

        _encoder_output = self.context_len_compressor(_output)

        # adding some nonlinearity
        _encoder_output = nn.functional.silu(_encoder_output) + _encoder_output
        _encoder_output = _encoder_output / torch.linalg.norm(_encoder_output)

        _encoder_output_do = self.dropout(_encoder_output)

        _output = self.context_len_decompressor(_encoder_output_do)

        _output = _output.view(num_batches, ctx_len, emb).contiguous()
        # adding some nonlinearity
        _output = nn.functional.silu(_output) + _output

        _output = self.token_embedding_decompressor(_output)

        return _encoder_output, _output

    def loss_fn(
        self,
        training_batch,
        output_batch,
        target_batch,
    ) -> torch.Tensor:
        """Returns the loss used for backpropagation"""

        _current_input_id, _current_input_tensor = training_batch
        _current_input_id = _current_input_id.clone().detach()
        _current_input_tensor = _current_input_tensor.clone().detach()

        encoder_output, output_tensor = output_batch
        

        if self.previous_input_id is None:
            self.previous_input_id = _current_input_id
            self.previous_input_tensor = _current_input_tensor
            # 
            self.previous_encoder_output = encoder_output.clone().detach()

        enc_sim = nn.functional.cosine_similarity(
            encoder_output, self.previous_encoder_output, dim=-1
        )
        # normalize encoder similarity to the range [0, 1]
        enc_sim = (enc_sim + 1.0) / 2.0
        # print(f"\n{enc_sim=}")

        reconstruction_loss = nn.functional.mse_loss(output_tensor, target_batch)
        contrastive_loss = torch.zeros_like(reconstruction_loss)

        # contrastive loss tensor, depending on whether the previous input and 
        # current one are similar or not
        _, _, tok_emb_dims = output_tensor.size()
        if _current_input_id == self.previous_input_id:
            # loss for similar texts is decreased by enc_sim
            # contrastive_loss = - torch.ones_like(reconstruction_loss) * enc_sim
            contrastive_loss = torch.zeros_like(reconstruction_loss)
        else:
            # loss for similar texts is increased by enc_sim
            contrastive_loss = tok_emb_dims * torch.ones_like(reconstruction_loss) * enc_sim
        
        # print(f"{contrastive_loss=}")

        loss = reconstruction_loss + contrastive_loss
        # print(f"{loss=}")

        self.previous_encoder_output = encoder_output.clone().detach()
        self.previous_input_id = _current_input_id
        self.previous_input_tensor = _current_input_tensor

        return loss

    def get_hyper_parameters(self) -> dict[str, Any]:
        """returns the hyperparamerters of the model"""
        return {
            "token_embedding_dim": self.token_embedding_dim,
            "max_context_length": self.max_context_length,
            "encoder_embeddings_size": self.encoder_embeddings_size,
            "multistep_lr_schedule": self.multistep_lr_schedule,
            "multistep_lr_gamma": self.multistep_lr_gamma,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
        }
