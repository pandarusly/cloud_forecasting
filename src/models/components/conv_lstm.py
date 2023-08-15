import numpy as np
import torch
import torch.nn as nn
import einops
from src.models.components.layers import ConvLSTMCell


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def schedule_sampling(eta, itr, batch_size, args):
    T, img_channel, img_height, img_width = args.in_shape
    zeros = np.zeros((batch_size,
                      args.aft_seq_length - 1,
                      img_height // args.patch_size,
                      img_width // args.patch_size,
                      args.patch_size ** 2 * img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (batch_size, args.aft_seq_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_height // args.patch_size,
                    img_width // args.patch_size,
                    args.patch_size ** 2 * img_channel))
    zeros = np.zeros((img_height // args.patch_size,
                      img_width // args.patch_size,
                      args.patch_size ** 2 * img_channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.aft_seq_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.aft_seq_length - 1,
                                  img_height // args.patch_size,
                                  img_width // args.patch_size,
                                  args.patch_size ** 2 * img_channel))

    real_input_flag = torch.FloatTensor(real_input_flag).permute(
        0, 1, 4, 2, 3).contiguous()  # b t c h w
    return eta, real_input_flag


def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size, seq_length, num_channels, img_height, img_width = img_tensor.shape
    patch_tensor = einops.rearrange(
        img_tensor, 'b s c (h p1) (w p2) -> b s (p1 p2 c)  h w', p1=patch_size, p2=patch_size)
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    batch_size, seq_length, channels,  patch_height, patch_width = patch_tensor.shape
    img_channels = channels // (patch_size * patch_size)
    img_tensor = einops.rearrange(
        patch_tensor, 'b s (p1 p2 c) h w-> b s c (h p1) (w p2) ', p1=patch_size, p2=patch_size)
    return img_tensor


class ConvLSTM_Model(nn.Module):
    r"""ConvLSTM Model

    Implementation of `Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting <https://arxiv.org/abs/1506.04214>`_.
    特点： 在训练过程中会逐渐的采样预测值作为输入。
    """
    # reverse_scheduled_sampling = 0

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(ConvLSTM_Model, self).__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                ConvLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                             configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def init_weights(self):
        self.apply(weights_init)

    def forward(self, frames, mask_true, **kwargs):

        frames = reshape_patch(frames, self.configs.patch_size)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for t in range(self.configs.pre_seq_length + self.configs.aft_seq_length - 1):
            # reverse schedule sampling for predrnn v2
            # if self.configs.reverse_scheduled_sampling == 1:
            #     if t == 0:
            #         net = frames[:, t]
            #     else:
            #         net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            #         # bchw = bchw * bchw + bchw * bchw
            # else:
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                    (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0])

            for i in range(1, self.num_layers):
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i])

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch,length, channel, height, width]
        next_frames = torch.stack(next_frames, dim=0).permute(
            1, 0, 2, 3, 4).contiguous()

        # if kwargs.get('return_loss', True):
        #     loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        # else:
        #     loss = None

        next_frames = reshape_patch_back(next_frames, self.configs.patch_size)
        return next_frames


if __name__ == '__main__':
    from omegaconf import OmegaConf

    # -----------model settings -----------
    configs = OmegaConf.structured(
        {
            "in_shape": (19, 3, 128, 128),
            "patch_size": 1,
            "pre_seq_length": 7,
            "aft_seq_length": 12,
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
    convlstm = ConvLSTM_Model(num_layers, num_hidden, configs,).cuda()
    # ----------forward settings ----------
    batch_x, batch_y = torch.randn(1, 7, 3, 128, 128).cuda(
    ), torch.randn(1, 12, 3, 128, 128).cuda()
    # TOTO: finish function : reshape_patch with einops
    eta = 1.0  # PredRNN variants
    num_updates = 0  # step number: initial num_updates = self._epoch * self.steps_per_epoch == 0 * steps_per_epoch
    ims = torch.cat([batch_x, batch_y], dim=1)
    eta, real_input_flag = schedule_sampling(
        eta, num_updates, ims.shape[0], configs)
    img_gen = convlstm(ims.cuda(), real_input_flag.cuda())
    print(img_gen.shape)
