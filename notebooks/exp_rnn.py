import hydra
import rootutils
rootutils.setup_root(".", indicator=".project-root", pythonpath=True)
import os
from omegaconf import OmegaConf
import torch
from src.utils import show_video_gif_multiple


config = r"logs/train/runs/2023-08-15_16-00-39/.hydra/config.yaml" 
config = OmegaConf.load(config)
# config.data.train_config.data_dir="./data/dataset/H8JPEG_valid"
# config.data.train_config.data_dir="./data/validate_croped"
# config.data.val_config.data_dir="./data/validate_croped"
# config.data.test_config.data_dir="./data/validate_croped"
# config.data.batch_size = 1
# config.data.batch_size_val = 4
# config.data.num_workers=0 
# config.data.pin_memory=True
# ckpt = r"./data/dataset/20230815-convlstm/step_1605_loss0.082.ckpt"  
ckpt = r"logs/train/runs/2023-08-15_16-00-39/checkpoints/step_14445_loss0.002.ckpt"  


data_module = hydra.utils.instantiate(config.data)
data_module.setup()
module = hydra.utils.instantiate(config.model)
module.load_state_dict(torch.load(ckpt,map_location="cpu")["state_dict"])
module = module.cuda()
module = module.eval()
from lightning.pytorch.trainer import Trainer
trainer = Trainer(
    accelerator="gpu",
    devices=1
)
trainer.test(model=module, datamodule=data_module, ckpt_path=ckpt)
# test/loss         │   0.0020648690406233072 

# from tqdm import tqdm
# batch_id = 0
# for batch_sample in tqdm(data_module.val_dataloader()):
#     # batch_sample = next(iter())
#     # from src.utils import show_video_line,show_video_gif_multiple,show_video_gif_single
#     # show_video_line(batch_sample["images"][0].numpy(), ncols=12, vmax=0.6, cbar=False, out_path="notebooks/samples.png", format='png', use_rgb=True)
#     batch_sample["images"] = batch_sample["images"].cuda()
#     with torch.no_grad():
#         img_gen, logs  = module.RNN_test_step(batch_sample, "val")
#     inputs=  batch_sample["images"][:,:2].cpu().detach().numpy()
#     trues =  batch_sample["images"][:,2:].cpu().detach().numpy()
#     preds = img_gen.cpu().numpy()

#     # print(inputs.shape)
#     # print(trues.shape)
#     # print(preds.shape)
#     # TODO: json 中有四个重复的值找下原因,这与地区相关
#     for example_idx in range(0, inputs.shape[0]):
#         show_video_gif_multiple(inputs[example_idx], trues[example_idx], preds[example_idx], use_rgb=True, out_path='./notebooks/example_{}_batch_{}.gif'.format(example_idx,batch_id))
#     batch_id+=1