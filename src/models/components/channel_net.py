

import segmentation_models_pytorch as smp

import warnings

import torch.nn as nn
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Channel_Net(nn.Module):  # TODO: add preprocessing Module
    def __init__(self, configs, **kwargs):
        super().__init__()
        # configs:
        #     name: "Unet"  # from segmentation_models_pytorch
        #     pre_seq_length: 7
        #     aft_seq_length: 12
        #     total_length: 19
        #     args: 
        #         encoder_name: "resnet34"
        #         encoder_weights: "imagenet"
        #         in_channels: 7*3
        #         classes: 12*3
        #         activation: "sigmoid"
        self.configs=configs
        self.unet = getattr(smp, self.configs.name)(**self.configs.args)

    def init_weights(self):
        self.unet.initialize()

    def forward(self, data):
        satimgs = data[:, : self.configs.pre_seq_length, ...]
        b, t, c, h, w = satimgs.shape
        satimgs = satimgs.reshape(b, t * c, h, w)
        
        outputs = self.unet(satimgs)
        if outputs.shape[-2:] == (h,w):
            outputs = resize(
                outputs,size =(h,w) ,mode="bilinear",align_corners=False
            ).reshape(b, self.configs.aft_seq_length, c, h, w)

        return outputs

if __name__ == "__main__":

    from omegaconf import OmegaConf
    import torch
    configs = OmegaConf.structured(
        {
            "name": "Unet",  # from segmentation_models_pytorch
            "pre_seq_length": 7,
            "aft_seq_length": 12,
            "total_length": 19,
            "args": {
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 7*2,
                "classes": 12*2,
                "activation": None,
            }
        }
    )
    model = Channel_Net(configs=configs).cuda()
    data = torch.randn(1,7,2,256,256).cuda()



    res = model(data)
    print(res.shape)
    print(res.min(),res.max())
