#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F


PI = 3.1415927410125732

def positional_encoding(x, scale=None, l=10):
    ''' Implements positional encoding on the given coordinates.
    
    Differentiable wrt to x.
    
    Args:
        x: torch.Tensor(n, dim)  - input coordinates
        scale: torch.Tensor(2, dim) or None 
            scale along the coords for normalization
            If None - scale inferred from x
        l: int - number of modes in the encoding
    Returns:
        torch.Tensor(n, dim + 2 * dim * l) - positional encoded vector.
    '''

    if scale is None:
        scale = torch.vstack([x.min(axis=0)[0], x.max(axis=0)[0]]).T

    x_normed = 2 * (x - scale[:, 0]) / (scale[:, 1] - scale[:, 0]) - 1

    if l > 0:
        sinuses = torch.concat([torch.sin( (2 ** p) * PI * x_normed) for p in range(l) ], axis=1)
        cosines = torch.concat([torch.cos( (2 ** p) * PI * x_normed) for p in range(l) ], axis=1)

        pos_enc = torch.concat([x_normed, sinuses, cosines], axis=1)
    else:
        pos_enc = x_normed
    return pos_enc


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        num_labels=0,
        num_pos_encodings=0,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        pe_dims = 3 + 6 * num_pos_encodings
        dims = [latent_size + pe_dims] + dims + [1 + num_labels]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        self.num_labels = num_labels
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= pe_dims

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
 
        self.sdf_activation = nn.ReLU()

        self.num_pos_encodings = num_pos_encodings
        self.pe_scale = torch.tensor([[-1, 1]] * 3).cuda()

    # input: N x (L+3)
    def forward(self, input):
        latent_vecs = input[:, :-3]
        xyz = positional_encoding(input[:, -3:], scale=self.pe_scale, l=self.num_pos_encodings)

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)

        x = input = torch.cat([latent_vecs, xyz], 1)

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # Apply sdf branch activation
        x[:, 0] = self.sdf_activation(x[:, 0])

        return x
