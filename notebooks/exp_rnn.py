import hydra
import rootutils
rootutils.setup_root(".", indicator=".project-root", pythonpath=True)
import os
from omegaconf import OmegaConf
import torch
from src.utils import show_video_gif_multiple


config = r"logs/train/runs/2023-08-15_16-00-39/.hydra/config.yaml" 
config = OmegaConf.load(config) 
config.data.batch_size_val = 1 
# ckpt = r"./data/dataset/20230815-convlstm/step_1605_loss0.082.ckpt"  
ckpt = r"logs/train/runs/2023-08-15_16-00-39/checkpoints/step_14445_loss0.002.ckpt"  


data_module = hydra.utils.instantiate(config.data)
data_module.setup()
# --------- model initialize and testing  ---------
module = hydra.utils.instantiate(config.model)
module.load_state_dict(torch.load(ckpt,map_location="cpu")["state_dict"])
module = module.cuda()
module = module.eval()
# from lightning.pytorch.trainer import Trainer
# trainer = Trainer(
#     accelerator="gpu",
#     devices=1
# )
# trainer.test(model=module, datamodule=data_module, ckpt_path=ckpt)
# test/loss         │   0.0020648690406233072 
# --------- plot making  ---------

from tqdm import tqdm
batch_id = 0

for batch_sample in tqdm(data_module.val_dataloader()):
    # from src.utils import show_video_line,show_video_gif_multiple,show_video_gif_single
    # show_video_line(batch_sample["images"][0].numpy(), ncols=12, vmax=0.6, cbar=False, out_path="notebooks/samples.png", format='png', use_rgb=True)
    batch_sample["images"] = batch_sample["images"].cuda() # 1 19 3 256 256
    batch_filenames = batch_sample["filenames"]  # 19 
    with torch.no_grad():
        img_gen, logs  = module.RNN_test_step(batch_sample, "val")

    # aasser batch size in valid is 1 -> t c h w 
    trues =  batch_sample["images"][0].cpu().detach().numpy() #(19, 3, 256, 256)
    preds = img_gen.cpu().numpy()[0] #torch.Size([18, 3, 256, 256])
 
    dir_name = os.path.join("./notebooks/plot",f"convlstm")
    img_name = os.path.basename(batch_filenames[0][0]).split(".")[0]
    os.makedirs(dir_name,exist_ok=True)
    index = config.model.net.configs.pre_seq_length
    # t c h w
    show_video_gif_multiple(trues[:index], trues[index:], preds[index-1:], use_rgb=True, out_path=f"{dir_name}/{img_name}_{batch_id}.gif")
    batch_id +=1
