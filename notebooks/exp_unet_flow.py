import os
import cv2
import hydra
import numpy as np
import rootutils
rootutils.setup_root(".", indicator=".project-root", pythonpath=True)
from omegaconf import OmegaConf
from PIL import Image 

from src.utils import show_video_gif_multiple

def pred_cv2(frame1, u, v, inter_method=cv2.INTER_LINEAR):
    h, w, c = frame1.shape
    row = np.arange(h)
    col = np.arange(w)
    colg, rowg = np.meshgrid(col, row)
    x = colg - u
    y = rowg - v
    ret = cv2.remap(frame1, x.astype(np.float32), y.astype(np.float32), inter_method)
    return ret


def read_pil(filename):
    return  np.array(Image.open(filename).convert('RGB'))


config = r"logs/train/runs/2023-08-16_19-47-21/.hydra/config.yaml" 
config = OmegaConf.load(config)
flow_type = "pyflow" # opencv_flow pyflow
config.data.val_config.get_optical_flow=flow_type
config.data.val_config.flow_version="by_step" 
flag = int(config.data.val_config.flow_version=="no_by_step")
config.data.batch_size_val = 1
ckpt = r"logs/train/runs/2023-08-16_19-47-21/checkpoints/step_12840_loss2.977.ckpt"  

data_module = hydra.utils.instantiate(config.data)
data_module.setup()
# --------- model initialize and testing  ---------
# module = hydra.utils.instantiate(config.model)
# module.load_state_dict(torch.load(ckpt,map_location="cpu")["state_dict"])
# module = module.cuda()
# module = module.eval()
# from lightning.pytorch.trainer import Trainer
# trainer = Trainer(
#     accelerator="gpu",
#     devices=1
# )
# trainer.test(model=module, datamodule=data_module, ckpt_path=ckpt)
# --------- flow ground truth experiment ---------

from tqdm import tqdm
batch_id = 0
for batch_sample in tqdm(data_module.val_dataloader()):
    batch_sample["images"] = batch_sample["images"].cuda()
    batch_filenames = batch_sample["filenames"] 
    trues = []
    for i in range(0,len(batch_filenames)):
        trues.append(read_pil(batch_filenames[i][0]).swapaxes(2,1).swapaxes(1,0)) # c h w 
    trues = np.stack(trues) # 18 3 256 256  落下最后一帧

    base_filename = batch_filenames[0][0] # only in batch_size == 1
    base_frame_rgb_array_255 = read_pil(base_filename) # h w c
    inputs = batch_sample["images"][0].cpu().detach().permute(0,2,3,1).numpy() # (18, 256, 256, 2)
    preds =[base_frame_rgb_array_255,]
    for i in range(len(inputs)):
        cur_frame = pred_cv2(base_frame_rgb_array_255,inputs[i][:,:,0],inputs[i][:,:,1])
        preds.append(cur_frame)
        if flag:
            base_frame_rgb_array_255 = cur_frame

    preds = np.stack(preds).swapaxes(3,2).swapaxes(2,1)[:-1] # (19-1, 3, 256, 256) 落下最后一帧
    dir_name = os.path.join("./notebooks/plot",f"bystep-{flag}-flowtype-{flow_type}")
    img_name = os.path.basename(batch_filenames[0][0]).split(".")[0]
    os.makedirs(dir_name,exist_ok=True)
    index = config.model.net.configs.pre_seq_length
    show_video_gif_multiple(trues[:index], trues[index:], preds[index:], use_rgb=True, out_path=f"{dir_name}/{img_name}_{batch_id}.gif")
    batch_id +=1
