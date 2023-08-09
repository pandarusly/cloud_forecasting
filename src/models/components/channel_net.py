from typing import Optional, Union

import argparse
import ast

import torch

import segmentation_models_pytorch as smp

from torch import nn


class Channel_Net(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams

        self.unet = getattr(smp, self.hparams.name)(**self.hparams.args)

        self.upsample = nn.Upsample(size=(128, 128))

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--name", type=str, default="densenet161")
        parser.add_argument(
            "--args",
            type=ast.literal_eval,
            default='{"encoder_name": "densenet161", "encoder_weights": "imagenet", "in_channels": 191, "classes": 80, "activation": "sigmoid}',
        )
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)

        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        n_preds = 0 if n_preds is None else n_preds

        satimgs = data["dynamic"][0][:, : self.hparams.context_length, ...]

        b, t, c, h, w = satimgs.shape

        satimgs = satimgs.reshape(b, t * c, h, w)

        dem = data["static"][0]
        clim = data["dynamic"][1][:, :, :5, ...]
        b, t, c, h2, w2 = clim.shape
        clim = (
            clim.reshape(b, t // 5, 5, c, h2, w2)
            .mean(2)[:, :, :, 39:41, 39:41]
            .reshape(b, t // 5 * c, 2, 2)
        )

        inputs = torch.cat((satimgs, dem, self.upsample(clim)), dim=1)

        return self.unet(inputs).reshape(b, self.hparams.target_length, 4, h, w), {}
