#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_lib.neural_blocks import *


class VQ_EMA(nn.Module):
    def __init__(
        self,
        independent_codebook,
        embedding_dim,
        num_embeddings,
        num_latents,
        decay=0.99,
        epsilon=1e-6,
    ):
        super().__init__()

        self._num_latents = num_latents
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._decay = decay
        self._epsilon = epsilon
        self._independent_codebook = independent_codebook

        latent_offset = torch.tensor(0)
        if independent_codebook:
            latent_offset = torch.arange(
                0, self._num_latents * self._num_embeddings, self._num_embeddings
            )
        else:
            self._num_latents = 1

        # initialize all codebooks identically
        embedding = 0.01 * torch.randn(self._embedding_dim, self._num_embeddings)
        embedding = torch.stack([embedding] * self._num_latents, dim=-1)

        self.register_buffer("_latent_offset", latent_offset, persistent=False)
        self.register_buffer("_embedding", embedding)
        self.register_buffer(
            "_dw",
            torch.zeros(self._embedding_dim, self._num_embeddings, self._num_latents),
        )
        self.register_buffer(
            "_cluster_size", torch.zeros(self._num_embeddings, self._num_latents)
        )

        # do this so VIM coc plays nicely with syntax
        self._latent_offset = self._latent_offset
        self._embedding = self._embedding
        self._dw = self._dw
        self._cluster_size = self._cluster_size

    def forward(self, input):
        """
        the encode path of the VQ
        """
        # squeeze HW together
        input_shape = input.shape
        input = input.view(input.shape[0], -1, input.shape[-2] * input.shape[-1])

        # calculate distances
        distances = self._embedding.unsqueeze(0) - input.unsqueeze(-2)
        distances = torch.linalg.norm(distances, dim=1)

        # get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)

        # get the encoding from the indices
        encoding = torch.zeros(distances.shape, device=input.device)
        encoding = encoding.scatter(1, encoding_indices.unsqueeze(1), 1)

        # quantize
        quantized = self.decode(encoding_indices)

        if self.training:
            # Laplce smoothing to handle division by 0
            n = torch.sum(self._cluster_size, -1).unsqueeze(-1) + 1e-6
            self._cluster_size = (
                (self._cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = None
            if self._independent_codebook:
                # update cluster size using ema
                self._cluster_size = self.ema(
                    self._cluster_size, torch.sum(encoding, dim=0)
                )
                # get innovation
                dw = torch.sum(encoding.unsqueeze(1) * input.unsqueeze(2), dim=0)

                # perplexity, divide by self._num_embeddings to make it num_embeddings agnostic
                # theoretical limit for this value is 1.0 for uniformly distributed codebook usage
                avg_probs = torch.mean(encoding, 0)
                perplexity = (
                    torch.exp(
                        -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=0)
                    ).detach()
                    / self._num_embeddings
                )

            else:
                # update cluster size using ema
                self._cluster_size = self.ema(
                    self._cluster_size,
                    torch.sum(torch.sum(encoding, dim=0), dim=-1, keepdim=True),
                )
                # get innovation
                dw = torch.sum(
                    torch.sum(encoding.unsqueeze(1) * input.unsqueeze(2), dim=0),
                    dim=-1,
                    keepdim=True,
                )

                # perplexity, divide by self._num_embeddings to make it num_embeddings agnostic
                # theoretical limit for this value is 1.0 for uniformly distributed codebook usage
                avg_probs = torch.mean(torch.mean(encoding, 0), -1)
                perplexity = (
                    torch.exp(
                        -torch.sum(
                            avg_probs * torch.log(avg_probs + 1e-10),
                            dim=0,
                            keepdim=True,
                        )
                    ).detach()
                    / self._num_embeddings
                )

            # update innovation using ema
            self._dw = self.ema(self._dw, dw / self._cluster_size.unsqueeze(0)).data

            # update embedding using innovation and ema
            self._embedding = self.ema(self._embedding, self._dw)

            # commitment loss
            commitment_loss = F.mse_loss(quantized.data, input)

            # straight through estimator
            quantized = input + (quantized - input).data

            # reshape quantized to input shape
            quantized = quantized.view(input_shape)
            encoding = encoding.view(input_shape[0], -1, *input_shape[-2:])

            return quantized, encoding, commitment_loss, perplexity
        else:
            # reshape quantized to input shape
            quantized = quantized.view(input_shape)
            encoding = encoding.view(input_shape[0], -1, *input_shape[-2:])
            return quantized, encoding, None, None

    def ema(self, value, update):
        return self._decay * value + (1 - self._decay) * update

    def decode(self, encoding_indices):
        """
        use offsets to select embeddings, credit https://github.com/andreaskoepf
        """
        shape = encoding_indices.shape

        # add offsets to indices
        encoding_indices = encoding_indices + self._latent_offset.unsqueeze(0)
        encoding_indices = encoding_indices.view(-1)

        # quantized
        quantized = self._embedding.permute(2, 1, 0).reshape(-1, self._embedding_dim)[
            encoding_indices
        ]
        quantized = (
            quantized.view(*shape, self._embedding_dim).permute(0, 2, 1).contiguous()
        )

        return quantized
