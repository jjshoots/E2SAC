#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from ai_lib.neural_blocks import *
from ai_lib.VQ import VQ


class VQVAE(nn.Module):
    """
    VQVAE model
    Basically the state encoder
    """

    def __init__(self, num_embedding, embedding_dim, independent_codebook=False):
        super().__init__()

        self.num_latents = 16
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim=embedding_dim)
        self.quantizer = VQ_EMA(
            independent_codebook=independent_codebook,
            embedding_dim=embedding_dim,
            num_embeddings=num_embedding,
            num_latents=self.num_latents,
        )

    def encode(self, input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def quantize(self, input):
        return self.quantizer(input)

    def encoding_indices_to_encoding(self, encoding_indices):
        """
        encoding indices is indices of the encodings
        encoding is one hot
        """
        shape = encoding_indices.shape
        encoding = torch.zeros(shape[0], self.num_embedding, *shape[-2:]).to(
            encoding_indices.device
        )
        encoding = encoding.scatter(1, encoding_indices.unsqueeze(1), 1.0)
        return encoding

    def encoding_to_quantized(self, encoding):
        """
        encoding is one hot
        quantized is the quantized embeddings
        """
        encoding_indices = torch.argmax(encoding, dim=2)
        return self.encoding_indices_to_quantized(encoding_indices)

    def encoding_indices_to_quantized(self, encoding_indices):
        """
        encoding indices is indices of the encodings
        quantized is the quantized embeddings
        """
        shape = encoding_indices.shape
        encoding_indices = encoding_indices.view(encoding_indices.shape[0], -1)
        quantized = self.quantizer.decode(encoding_indices)

        return quantized.view(shape[0], -1, *shape[-2:])


class Encoder(nn.Module):
    """
    Encoder for VQVAE
    """

    def __init__(self, embedding_dim):
        super().__init__()

        _channels_description = [3, 64, 64, 128, 128, embedding_dim]
        _kernels_description = [3, 3, 3, 3, 1]
        _pooling_description = [2, 2, 2, 2, 0]
        _activation_description = ["lrelu"] * (len(_kernels_description) - 1) + [
            "sigmoid"
        ]
        self.model = Neural_blocks.generate_conv_stack(
            _channels_description,
            _kernels_description,
            _pooling_description,
            _activation_description,
        )

    def forward(self, input):
        output = self.model(input)
        return output


class Decoder(nn.Module):
    """
    Decoder for VQVAE
    """

    def __init__(self, embedding_dim):
        super().__init__()

        _channels = [embedding_dim, 512, 256, 128, 64]
        _kernels = [4, 4, 4, 4]
        _padding = [1, 1, 1, 1]
        _stride = [2, 2, 2, 2]
        _activation = ["lrelu", "lrelu", "lrelu", "lrelu"]
        self.unsqueeze = Neural_blocks.generate_deconv_stack(
            _channels, _kernels, _padding, _stride, _activation
        )

        # the final image, 3x64x64
        _channels = [64, 32, 3]
        _kernels = [3, 1]
        _pooling = [0, 0]
        _activation = ["lrelu", "sigmoid"]
        self.regenerate = Neural_blocks.generate_conv_stack(
            _channels, _kernels, _pooling, _activation
        )

    def forward(self, input):
        output = self.unsqueeze(input)
        output = self.regenerate(output)

        return output
