import torch
import segmentation_models_pytorch as smp

from torch import nn
from src.models.rnn import RNNModule

class UnetModule(RNNModule):

    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)


    def forward():
         pass
         
    def RNN_train_step(self, batch, step_name: str):
            logs = {}
            self.model.configs.total_length
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

    # if batch_idx < self.hparams.n_log_batches:
    #         self.log_viz(all_viz, batch, batch_idx)
