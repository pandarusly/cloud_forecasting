from torch import nn
import torch.nn.functional as F


class MaskedLoss(nn.Module):
    def __init__(self, distance_type="L2"):
        super(MaskedLoss, self).__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask):
        assert preds.shape == targets.shape
        predsmasked = preds * mask
        targetsmasked = targets * mask

        if self.distance_type == "L2":
            return F.mse_loss(predsmasked, targetsmasked, reduction="sum") / (
                (mask > 0).sum() + 1
            )
        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked, targetsmasked, reduction="sum") / (
                (mask > 0).sum() + 1
            )


LOSSES = {"masked": MaskedLoss}


class BaseLoss(nn.Module):
    def __init__(self, setting: dict):
        super().__init__()

        self.distance = LOSSES[setting["name"]](**setting["args"])

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        targs = batch["dynamic"][0][:, -preds.shape[1] :, ...]
        masks = batch["dynamic_mask"][0][:, -preds.shape[1] :, ...]

        dist = self.distance(preds, targs, masks)

        logs["distance"] = dist

        loss = dist

        logs["loss"] = loss

        return loss, logs


def setup_loss(args):
    return BaseLoss(args)
