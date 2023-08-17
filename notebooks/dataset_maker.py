import json
import random

import tqdm
from src.data.components.cloud_sequence import BaseCloudRGBSequenceDataset
random.seed(42)

def make_1024():
    random.seed(42)
    step = 19
    Intinterval = 19
    crop_size = (512, 512)

    split_path = f"data/ceshi_maker_{step}_interval_{Intinterval}.json"
    data_dir = "data/DLDATA/H8JPEG_valid" 

    dataset = BaseCloudRGBSequenceDataset(
        data_dir=data_dir,
        use_transform=True,
        get_optical_flow="none",
        split_path=split_path,
        step=step,
        Intinterval=Intinterval,
        crop_size=crop_size,
    )

    with open(split_path, "r") as json_file:
        data = json.load(json_file)
        image_list = sorted(data)

    length = len(image_list)
    val_length = int(0.8*length)
    print(f"共有 {length} 组不重叠的时间序列")
    train_images = image_list[:val_length]
    val_images = image_list[val_length:]

    print(
        f"training time series length is {len(train_images)}\n",
        f"validate time series length is {len(val_images)} \n"
    )
    with open(split_path.replace(".json", "_train.json"), "w") as f:
        f.write(json.dumps(train_images))
    with open(split_path.replace(".json", "_val.json"), "w") as f:
        f.write(json.dumps(val_images))

    val_dataset = BaseCloudRGBSequenceDataset(
        data_dir=data_dir,
        use_transform=True,
        get_optical_flow="none",
        split_path=split_path.replace(".json", "_val.json"),
        step=step,
        Intinterval=Intinterval,
        crop_size=crop_size,
        div_255=False
    )

    PLot = False
    augment_save_dir = "data/test_croped"
 
    for i in range(len(val_dataset)):
        data = val_dataset[i]
        if PLot:
            val_dataset.plot_frames(data, out_directory=augment_save_dir)
        else:
            val_dataset.make_dataset(
                data,
                out_directory=augment_save_dir,
                random_string="valid",
            )

if __name__ == "__main__":
    make_1024()
