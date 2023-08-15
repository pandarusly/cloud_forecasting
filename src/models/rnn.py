from abc import ABC
from typing import Any, Dict
from omegaconf import OmegaConf
import torch
# from mmcv.cnn.utils import revert_sync_batchnorm

from lightning import LightningModule
from src.models.components.conv_lstm import schedule_sampling
# from lightnings.utils.binary import ConfuseMatrixMeter
# from pytorch_lightning.callbacks import ModelCheckpoint
# from mmseg.core import add_prefix
# from mmseg.models import build_segmentor
from src.models.vendor import build_optimizer, build_scheduler
from torch import nn

class RNNModule(LightningModule, ABC):
    def __init__(
            self,
            net: nn.Module,
            loss: nn.Module,
            hparams,  # OmegaConf
            CKPT=False,
            **kwargs
    ):
        super().__init__()

        # hparams contains "TRAIN"
        self.save_hyperparameters(hparams, logger=False)
        self.lr = self.hparams.TRAIN.BASE_LR

        self.model = net
        self.model.init_weights()
        self.eta = 1.0

        self.loss = loss

        # example_input_array=(1, 6, 256, 256)
        # self.example_input_array = torch.randn(*example_input_array)

        if CKPT:
            self._finetue(CKPT)

    def _finetue(self, ckpt_path):
        print("-" * 30)
        print("locate new momdel pretrained {}".format(ckpt_path))
        print("-" * 30)
        pretained_dict = torch.load(ckpt_path)["state_dict"]
        self.load_state_dict(pretained_dict)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self.parameters())
        scheduler = build_scheduler(self.hparams, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.TRAIN.INTERVAL,
                "monitor": self.hparams.TRAIN.MONITOR
            },
        }

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     # print("\ncurrent_epoch ", self.current_epoch)
    #     # print("global_step", self.global_step)
    #     scheduler.step(
    #         epoch=self.current_epoch
    #     )  # timm's scheduler need the epoch value

    def RNN_test_step(self, batch, step_name: str):
        logs = {}
        ims = batch["images"]
        # batch_x, batch_y = batch
        # ims = torch.cat([batch_x, batch_y], dim=1)

        mask_input = self.model.configs.pre_seq_length

        _, img_channel, img_height, img_width = self.model.configs.in_shape

        real_input_flag = torch.zeros(
            (ims.shape[0],
             self.model.configs.total_length - mask_input - 1,
             self.model.configs.patch_size ** 2 * img_channel,
             img_height // self.model.configs.patch_size,
             img_width // self.model.configs.patch_size)).to(ims.device)

        img_gen = self.forward(frames=ims, mask_true=real_input_flag)
        loss = self.loss(img_gen, ims[:, 1:])
        logs[f"{step_name}/loss"] = loss

        return img_gen, logs

    def RNN_train_step(self, batch, eta=1.0, num_updates=0):
        logs = {}
        ims = batch["images"]
        # step number: initial num_updates = self._epoch * self.steps_per_epoch == 0 *
        # eta = 1.0  # PredRNN variants
        # batch_x, batch_y = torch.randn(1, 7, 3, 128, 128).cuda(
        # ), torch.randn(1, 12, 3, 128, 128).cuda()
        # filenames=  batch["filenames"]
        # batch_x, batch_y = batch
        # ims = torch.cat([batch_x, batch_y], dim=1)
        eta, real_input_flag = schedule_sampling(
            eta, num_updates, ims.shape[0], self.model.configs)
        real_input_flag = real_input_flag.to(ims.device)

        self.eta = eta
        img_gen = self.forward(frames=ims, mask_true=real_input_flag)
        loss = self.loss(img_gen, ims[:, 1:])

        logs["train/loss"] = loss
        logs["train/eta"] = eta

        return loss, logs

    def training_step(self, batch: Dict[str, Any], batch_idx: int):

        loss, logs = self.RNN_train_step(batch, self.eta, self.global_step)
        self.log_dict(logs, prog_bar=True, sync_dist=True,
                      on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):

        img_gen, logs = self.RNN_test_step(batch, "val")
        self.log_dict(logs, prog_bar=True, sync_dist=True,
                      on_step=False, on_epoch=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int):

        img_gen, logs = self.RNN_test_step(batch, "test")
        self.log_dict(logs, prog_bar=True, sync_dist=True,
                      on_step=False, on_epoch=True)


if __name__ == "__main__":
    from src.models.components.conv_lstm import ConvLSTM_Model
    # -----------model settings -----------
    configs = OmegaConf.structured(
        {
            "in_shape": (19, 3, 65, 65),
            "patch_size": 1,
            "pre_seq_length": 7,
            "aft_seq_length": 12,
            "total_length": 19,
            "filter_size": 5,
            "stride": 1,
            "layer_norm": 0,
            # scheduled sampling
            "scheduled_sampling": 1,
            "sampling_stop_iter": 50000,
            "sampling_start_value": 1.0,
            "sampling_changing_rate": 0.00002,
        }
    )
    num_hidden = [128, 128, 128, 128]
    num_layers = len(num_hidden)
    # num_layers, num_hidden, configs, **kwargs
    convlstm = ConvLSTM_Model(num_layers, num_hidden, configs,)

    # ----------task settings------------------------
    task = RNNModule(
        net=convlstm,
        loss=torch.nn.MSELoss(),
        hparams=OmegaConf.load("configs/model/convlstm.yaml").hparams
    )
    # # ----------forward settings ----------
    batch_x, batch_y = torch.randn(
        1, configs.pre_seq_length, 3, configs.in_shape[-1], configs.in_shape[-1]), torch.randn(1, configs.aft_seq_length, 3, configs.in_shape[-1], configs.in_shape[-1])
    batch = (batch_x, batch_y)
    batch = dict(
        images=torch.randn(1, configs.total_length, 3,
                           configs.in_shape[-1], configs.in_shape[-1])
    )

    loss = task.training_step(batch, 0)
    print(f"train loss {loss}")
    task.validation_step(batch, 0)
    task.test_step(batch, 0)

    # # TOTO: finish function : reshape_patch with einops
    # eta = 1.0  # PredRNN variants
    # num_updates = 0  # step number: initial num_updates = self._epoch * self.steps_per_epoch == 0 * steps_per_epoch
    # ims = torch.cat([batch_x, batch_y], dim=1)
    # eta, real_input_flag = schedule_sampling(
    #     eta, num_updates, ims.shape[0], configs)
    # img_gen = convlstm(ims.cuda(), real_input_flag.cuda())
    # print(img_gen.shape)
