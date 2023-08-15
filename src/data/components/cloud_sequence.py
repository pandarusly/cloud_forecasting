import json
import os
import random
import re
from datetime import datetime, timedelta
import string
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
import pickle

from imgaug import augmenters as iaa
from src.data.components.flow_utils import (
    get_optical_flow_pyflow,
    get_opticalflow_cv2,
    write_flo_file,
    read_flo_file,
)

random.seed(42)


def get_time(parameter="total"):
    def deco_func(f):
        def wrapper(*arg, **kwarg):
            s_time = time.time()
            res = f(*arg, **kwarg)
            e_time = time.time()
            print("{} time use {}s".format(parameter, e_time - s_time))
            return res

        return wrapper

    return deco_func


# TODO:fixed bug fixed:The issue seems to be with the Intel OpenMP runtime library (libiomp5md.dll) being loaded multiple times or conflicting with another OpenMP runtime library. This can result in performance degradation or incorrect behavior in your program.
# set KMP_DUPLICATE_LIB_OK=TRUE in Environment Variable Workaround
# TODO 学习pyflow的Cython打包流程: https://github.com/pathak22/pyflow


def save_image_with_pil(image_data, output_file_path):
    """
    Save an image using the Python Imaging Library (PIL).

    Args:
        image_data (numpy.ndarray): Numpy array containing the image data.
        output_file_path (str): Path to the output file where the image will be saved.
    """
    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(image_data.astype(np.uint8))

    # Save the PIL image to the specified file path
    pil_image.save(output_file_path)

    # print("Image saved to:", output_file_path)


def get_strdatetime_filename_mapping_and_list(directory_path):
    # Regular expression pattern to match YYYYMMDD and HHMM in the filename
    pattern = r"(\d{8})_(\d{4})"

    strdatetime_list = []
    datatimes_nums = set()
    # Create a dictionary to map datetime objects to filenames
    datetime_filename_mapping = {}
    croped_region_nums = set()
    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        # Use regular expression to find the date and time in the filename
        match = re.search(pattern, filename)
        croped_preffix = os.path.basename(filename).split("_")[0]  # 8
        croped_region_nums.add(croped_preffix)
        if match:
            date = match.group(1)
            time = match.group(2)
            strdatetime_list.append(croped_preffix + "_" + date + time)
            datatimes_nums.add(date + time)
            datetime_filename_mapping[croped_preffix +
                                      "_" + date + time] = filename
    croped_region_nums = len(list(croped_region_nums))
    datatimes_nums = len(list(datatimes_nums))

    return (
        sorted(strdatetime_list),
        datetime_filename_mapping,
        croped_region_nums,
        datatimes_nums,
    )


def generate_subsequent_intervals(
    start_index_for_datetime_list: int,
    strdatetime_list: List[str],
    datetime_filename_mapping: Dict[str, str],  # Use Dict instead of map
    Intinterval: int = 19,
):
    subsequent_intervals = []
    if start_index_for_datetime_list + Intinterval > len(strdatetime_list):
        return subsequent_intervals

    tinterval = timedelta(minutes=10)

    current_strdatetime = strdatetime_list[start_index_for_datetime_list]
    current_datetime = datetime.strptime(
        current_strdatetime.split("_")[-1], "%Y%m%d%H%M"
    )
    croped_preffix = current_strdatetime.split("_")[0]
    for _ in range(Intinterval):
        filename = datetime_filename_mapping.get(
            croped_preffix + "_" + current_datetime.strftime("%Y%m%d%H%M")
        )
        if filename is None:
            break  # Stop if no corresponding filename
            # return subsequent_intervals
        subsequent_intervals.append(filename)
        current_datetime += tinterval

    return subsequent_intervals


def make_dataset_json(
    Intinterval, step, directory_path=r"data/DLDATA/H8JPEG_valid", split_path=None
):
    # Intinterval = 19
    # step = 0
    # OVERLAP = Intinterval - step

    # Replace 'path_to_directory' with the actual path to your directory containing the pictures
    (
        strdatetime_list,
        datetime_filename_mapping,
        croped_region_nums,
        datatimes_nums,
    ) = get_strdatetime_filename_mapping_and_list(directory_path)

    print(
        f"A total of {len(strdatetime_list)} image, and has {croped_region_nums} regions; {datatimes_nums} datatimes"
    )
    datasets = []

    for i in range(0, croped_region_nums):
        time_seris_num = 0
        for j in range(0, datatimes_nums, step):
            subsequent_intervals = generate_subsequent_intervals(
                j, strdatetime_list, datetime_filename_mapping, Intinterval=Intinterval
            )
            if len(subsequent_intervals) == Intinterval:
                datasets.append(subsequent_intervals)

            time_seris_num += 1
        # len(datatimes_nums)
        subsequent_intervals = generate_subsequent_intervals(
            j + datatimes_nums,
            strdatetime_list,
            datetime_filename_mapping,
            Intinterval=Intinterval,
        )
        if len(subsequent_intervals) == Intinterval:
            datasets.append(subsequent_intervals)

    if split_path:
        json_path = split_path
        # json_path = (
        #     f"data/dataset_step_{step}_interval_{Intinterval}.json"
        #     if split_path is None
        #     else split_path
        # )
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        print(
            f"only {time_seris_num} time series is generated wihch has {croped_region_nums} regions ,and {len(datasets)} time series is valid , length is {len(datasets[0])} "
        )
        with open(json_path, "w") as f:
            f.write(json.dumps(datasets))
    return datasets


def generate_random_string(length):
    return str(datetime.now().hour) + \
        str(datetime.now().minute)+str(datetime.now().second) +\
        "".join(random.choice(string.ascii_letters) for _ in range(length))

    # return


# TODO:fixed tar -czvf x.tar.gz  x --- 使用了gzip进行了有损压缩。会影响JPG的质量。
class BaseCloudRGBSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split_path=None,
        use_transform=False,
        get_optical_flow=False,
        Intinterval=19,
        step=1,
        input_frames_num=6,
        crop_size=(1024, 1024),
    ):
        if not os.path.exists(data_dir):
            raise FileExistsError(f"data_dir {data_dir} is not exist")
        self.data_dir = data_dir
        self.use_transform = use_transform
        self.interval = Intinterval
        self.input_frames_num = input_frames_num
        self.use_optical_flow = False
        self.split_path = split_path
        self.crop_size = crop_size

        if not os.path.exists(split_path):

            self.image_list = sorted(self._make_dataset_json(
                Intinterval=Intinterval, step=step, split_path=split_path
            ))
            print(
                f"split_path is not specified, reading from {self.data_dir}, got {len(self.image_list)} samples")
        else:
            with open(self.split_path, "r") as json_file:
                data = json.load(json_file)
            self.image_list = sorted(data)
            print(
                f"load samples from {split_path}, got {len(self.image_list)} samples")

        if get_optical_flow == "opencv_flow":
            self.use_optical_flow = True
            self._get_optical_flow = get_opticalflow_cv2
        elif get_optical_flow == "pyflow":
            self.use_optical_flow = True
            self._get_optical_flow = get_optical_flow_pyflow

        if use_transform:
            self.transform = iaa.Sequential(
                [
                    iaa.CropToFixedSize(
                        width=self.crop_size[0], height=self.crop_size[1]
                    ),  # 裁剪到1024
                    # 选择2到4种方法做变换
                    # iaa.SomeOf(
                    #     (2, 4),
                    #     [
                    #         iaa.Fliplr(0.5),  # 对50%的图片进行水平镜像翻转
                    #         iaa.Flipud(0.5),  # 对50%的图片进行垂直镜像翻转
                    #         # iaa.OneOf(
                    #         #     [
                    #         #         iaa.Affine(rotate=(-10, 10), scale=(1.1, 1.2)),
                    #         #         iaa.Affine(
                    #         #             translate_px={
                    #         #                 "x": (-10, 10),
                    #         #                 "y": (-7, -13),
                    #         #             },
                    #         #             scale=(1.1),
                    #         #         ),
                    #         #     ]
                    #         # ),
                    #         # Blur each image with varying strength using
                    #         # gaussian blur,
                    #         # average/uniform blur,
                    #         # median blur .
                    #         # iaa.OneOf(
                    #         #     [
                    #         #         iaa.GaussianBlur((0.2, 0.7)),
                    #         #         iaa.AverageBlur(k=(1, 3)),
                    #         #         iaa.MedianBlur(k=(1, 3)),
                    #         #     ]
                    #         # ),
                    #         # Sharpen each image, overlay the result with the original
                    #         # iaa.Sharpen(alpha=1.0, lightness=(0.7, 1.3)),
                    #         # Same as sharpen, but for an embossing effect.
                    #         # iaa.Emboss(alpha=(0, 0.3), strength=(0, 1)),
                    #         # 添加高斯噪声
                    #         # iaa.AdditiveGaussianNoise(
                    #         #     loc=0, scale=(0.01 * 255, 0.05 * 255)
                    #         # ),
                    #         # iaa.Grayscale((0.2, 0.7)),
                    #         # # iaa.Invert(0.05, per_channel=True),  # invert color channels
                    #         # # Add a value of -10 to 10 to each pixel.
                    #         # iaa.Add((-10, 10), per_channel=0.5),
                    #         # iaa.OneOf(
                    #         #     [
                    #         #         iaa.AddElementwise((-20, 20)),
                    #         #         iaa.MultiplyElementwise((0.8, 1.2)),
                    #         #         iaa.Multiply((0.7, 1.3)),
                    #         #     ]
                    #         # ),
                    #         # Improve or worsen the contrast of images.
                    #         # iaa.ContrastNormalization((0.7, 1.3)),
                    #     ],
                    #     # do all of the above augmentations in random order
                    #     random_order=True,
                    # ),
                ],
                random_order=True,
            )

    def __len__(self):
        return len(self.image_list)

    # @get_time("__getitem__")
    def __getitem__(self, idx) -> torch.Tensor:  # t c h w
        image_paths = [
            os.path.join(self.data_dir, basename) for basename in self.image_list[idx]
        ]

        frames = []
        filenames = []
        for image_path in image_paths:
            frames.append(np.array(Image.open(image_path).convert("RGB")))
            filenames.append(image_path)
        # Apply augmentation using albumentations
        # Time Series Data Augmentation for Deep Learning: A Survey
        # https://blog.csdn.net/weixin_47954807/article/details/114908771

        if self.use_transform:
            transform_deterministic = self.transform.to_deterministic()
            augmented_frames = [
                transform_deterministic.augment_image(frame) for frame in frames
            ]
            frames = augmented_frames

        if self.use_optical_flow:
            frames = self._get_optical_flow(frames)
            filenames = filenames[:-1]

        # Convert frames list to a tensor
        frames = np.stack(frames)
        frames = torch.tensor(frames, dtype=torch.float32).permute(
            0, 3, 1, 2
        )  # t h w c -> t c h w

        data = dict(images=frames, filenames=filenames)
        return data

    @get_time("plot_frames")
    def plot_frames(self, data: Dict, out_directory):
        tensors, filenames = data["images"], data["filenames"]

        # Create the output directory if it doesn't exist
        os.makedirs(out_directory, exist_ok=True)

        for i in range(tensors.shape[0]):
            tensor = tensors[i]
            filename = filenames[i]
            filename = os.path.basename(filename)

            if self.use_optical_flow:
                image_np_u = tensor.numpy()[0, :, :]
                image_np_v = tensor.numpy()[1, :, :]

                # Plot the image
                plt.figure(figsize=(20, 20))
                plt.subplot(1, 2, 1)
                plt.imshow(image_np_u, cmap="hsv")
                plt.axis("off")  # Turn off axis labels and ticks
                plt.title("u")
                plt.subplot(1, 2, 2)
                plt.imshow(image_np_v, cmap="hsv")
                plt.axis("off")  # Turn off axis labels and ticks
                plt.title("u")

                # Save the plot to the output directory
                output_path = os.path.join(out_directory, f"{filename}")
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

            else:
                # Convert the tensor to a numpy array
                # chw to hwc
                image_np = (
                    tensor.permute(1, 2, 0).numpy() / 255.0
                    if tensor.max() > 1
                    else tensor.permute(1, 2, 0).numpy()
                )

                # Plot the image
                plt.figure(figsize=(20, 20))
                plt.imshow(image_np)
                plt.axis("off")  # Turn off axis labels and ticks
                plt.title(filename)

                # Save the plot to the output directory
                output_path = os.path.join(out_directory, f"{filename}")
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

    def _make_dataset_json(self, Intinterval, step, split_path):
        directory_path = self.data_dir
        return make_dataset_json(
            Intinterval, step, directory_path, split_path
        )

    @get_time("make_dataset")
    def make_dataset(
        self, data, out_directory, random_string
    ):  # Get current timestamp):
        tensors, filenames = data["images"], data["filenames"]

        # Create the output directory if it doesn't exist
        os.makedirs(out_directory, exist_ok=True)

        for i in range(tensors.shape[0]):
            tensor = tensors[i]
            filename = filenames[i]

            if self.use_optical_flow:
                flow_data = tensor.permute(1, 2, 0).numpy()
                # image_np_u = flow_data[:, :, 0]
                # image_np_v = flow_data[:, :, 1]
                base_name = os.path.basename(filename).split(".")[0]
                output_path = os.path.join(
                    out_directory, f"{random_string}_{base_name}.flo"
                )
                write_flo_file(filename=output_path, flow_data=flow_data)

            else:
                # Convert the tensor to a numpy array
                # chw to hwc
                # no need to normalize here
                assert tensor.max() > 1
                image_np = tensor.permute(1, 2, 0).numpy()
                base_name = os.path.basename(filename).split(".")[0]
                output_path = os.path.join(
                    out_directory, f"{random_string}_{base_name}.jpg"
                )
                save_image_with_pil(image_np, output_path)

    def make_pkldataset(self, data_list, out_dir, pre_slen=10, aft_slen=10, split="train"):
        """ make vedios pkl for more speed reading.

        Args:
            data_list (list): [__getitem__,return types,data["images"], data["filenames"]]
            out_directory (str): _description_
            random_string (str): _description_
        Returns:
            Vedios (dict): B T C H W: x_region_filename, y_region_filename
        """

        videos = []
        dataset = {}
        for data in data_list:
            # vedio, _ = data["images"].numpy(), data["filenames"]
            vedio = data["images"].numpy()

            videos.append(vedio)
        # stack video frames from each folder
        data = np.stack(videos)   # btchw
        data_x, data_y = data[:, :pre_slen], data[:, pre_slen:]
        # if the data is in [0, 255], rescale it into [0, 1]
        if data.max() > 1.0:
            data = data.astype(np.float32) / 255.0
        dataset['X_' + split], dataset['Y_' + split] = data_x, data_y

        # save as a pkl file
        out_file = os.path.join(out_dir, f"{split}_cloud.pkl")
        with open(out_file, 'wb') as f:
            pickle.dump(dataset, f)

        return out_file


#  opencv is slower than pil
#  TODO:
class CloudFlowSequenceDataset(BaseCloudRGBSequenceDataset):
    def __init__(self, use_transform=False, Intinterval=19, step=19, *args, **kwargs):
        super().__init__(
            Intinterval=Intinterval - 1,
            step=max(1, step - 1),
            use_transform=use_transform,
            *args,
            **kwargs,
        )

        self.use_optical_flow = True

    def __getitem__(self, idx) -> torch.Tensor:  # t c h w
        image_paths = [
            os.path.join(self.data_dir, basename) for basename in self.image_list[idx]
        ]

        frames = []
        filenames = []

        for image_path in image_paths:
            image_path = image_path.replace(".jpg", ".flo")
            if os.path.exists(image_path):
                frames.append(read_flo_file(image_path))
                filenames.append(image_path)
        # Apply augmentation using albumentations
        # Time Series Data Augmentation for Deep Learning: A Survey
        # https://blog.csdn.net/weixin_47954807/article/details/114908771

        if self.use_transform:
            transform_deterministic = self.transform.to_deterministic()
            augmented_frames = [
                transform_deterministic.augment_image(frame) for frame in frames
            ]
            frames = augmented_frames

        if len(frames) == 0:
            raise ValueError(f"directory {self.data_dir} is not qualifie")
        # Convert frames list to a tensor
        frames = np.stack(frames)
        frames = torch.tensor(frames, dtype=torch.float32).permute(
            0, 3, 1, 2
        )  # t h w c -> t c h w

        data = dict(images=frames, filenames=filenames)
        return data


def process_dataset_maker(PLot=False, flow_type="opencv_flow"):
    random.seed(42)
    step = 1  # TODO: this step should be equal to interval , otherwise will have same name
    Intinterval = 19
    crop_size = (256, 256)
    crop_times = 4
    split_path = f"data/crop_dataset_step_{step}_interval_{Intinterval}.json"
    data_dir = "data/DLDATA/H8JPEG_valid"

    augment_save_dir = (
        "data/CLip_H8-{}_flow_type-{}_PLot{}".format(
            Intinterval, flow_type, PLot
        )
    )

    dataset = BaseCloudRGBSequenceDataset(
        data_dir=data_dir,
        use_transform=True,
        get_optical_flow=flow_type,
        split_path=split_path,
        step=step,
        Intinterval=Intinterval,
        crop_size=crop_size,
    )

    # for _ in tqdm(range(crop_times), position=1, colour="red"):
    #     random_string = generate_random_string(8)
    #     for i in tqdm(
    #         range(len(dataset)), position=2, colour="blue", desc="dataset making!!!"
    #     ):
    #         data = dataset[i]

    #         if PLot:
    #             dataset.plot_frames(data, out_directory=augment_save_dir)
    #         else:
    #             dataset.make_dataset(
    #                 data,
    #                 out_directory=augment_save_dir,
    #                 random_string=random_string,
    #             )

    # return augment_save_dir


def process_flo_dataset(flow_type="opencv_flow", data_dir=None):
    random.seed(42)
    step = 19
    Intinterval = 19
    crop_size = (256, 256)
    # crop_times = 1
    split_path = f"data/flow_dataset_step_{step}_interval_{Intinterval}.json"
    data_dir = "data/DLDATA/H8JPEG_valid" if not data_dir else data_dir
    PLot = False
    # flow_type = "opencv_flow"  # opencv_flow pyflow none
    augment_save_dir = (
        "data/DLDATA/H8JPEG_valid_aug_Intinterval-{}_flow_type-{}_PLot{}".format(
            Intinterval, flow_type, PLot
        )
    )

    # data_dir = augment_save_dir
    # augment_save_dir = f"data/DLDATA/H8JPEG_valid_aug_{crop_size[0]}"

    dataset = CloudFlowSequenceDataset(
        data_dir=data_dir,
        use_transform=False,
        split_path=split_path,
        step=step,
        Intinterval=Intinterval,
        crop_size=crop_size,
    )

    for i in range(0, len(dataset)):
        tensors, filenames = dataset[i]["images"], dataset[i]["filenames"]
        arrays_sequence = tensors.permute(0, 2, 3, 1).numpy()
        img_path = os.path.join(
            "data\dataset\H8JPEG_valid_aug_Intinterval-19_flow_type-none_PLotFalse",
            os.path.basename(filenames[0]).replace(".flo", ".jpg"),
        )
        initial_image = np.array(Image.open(img_path).convert("RGB"))
        # 外推的时间步长，根据实际情况进行调整
        for t in range(arrays_sequence.shape[0]):
            flow = arrays_sequence[t]
            # 外推图像坐标
            x_coords, y_coords = np.meshgrid(
                np.arange(initial_image.shape[1]), np.arange(
                    initial_image.shape[0])
            )
            x_coords_ext = x_coords + t * flow[:, :, 0]
            y_coords_ext = y_coords + t * flow[:, :, 1]
            # 使用双线性插值重建图像
            reconstructed_image = cv2.remap(
                initial_image,
                x_coords_ext.astype(np.float32),
                y_coords_ext.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
            )

            cv2.imwrite(
                "data/reconstructed_image_{}.jpg".format(
                    t), reconstructed_image
            )


def process_pkldataset_maker(save_dir="data", plot=True):
    random.seed(42)
    step = 19
    Intinterval = 19
    crop_size = (256, 256)
    crop_times = 1

    split_path = f"data/demo_dataset_step_{step}_interval_{Intinterval}.json"
    flow_type = "none"

    dataset = BaseCloudRGBSequenceDataset(
        data_dir="data/DLDATA/H8JPEG_valid",
        use_transform=True,
        get_optical_flow=flow_type,
        split_path=split_path,
        step=step,
        Intinterval=Intinterval,
        crop_size=crop_size,
    )

    pre_slen = 6

    splits = []
    for _ in tqdm(range(crop_times), position=1, colour="red"):
        random_string = generate_random_string(8)
        vedios = []
        for i in tqdm(
            range(len(dataset)), position=2, colour="blue", desc="dataset making!!!"
        ):
            if plot:  # TODO: plot frame sequence is not match .pkl store sequene
                dataset.plot_frames(
                    dataset[i], out_directory=f"{save_dir}/pkl_images")
            vedios.append(dataset[i])
        dataset.make_pkldataset(
            vedios, save_dir, pre_slen=pre_slen, split=f"{random_string}_trian_{crop_size[0]}_interval-{Intinterval}_pre_slen-{pre_slen}"
        )
        splits.append(
            f"{random_string}_trian_{crop_size[0]}_interval-{Intinterval}_pre_slen-{pre_slen}")

    return splits


def tranvaltest_json_split():
    random.seed(42)
    step = 19
    Intinterval = 19
    crop_size = (256, 256)

    split_path = f"data/total_dataset_step_{step}_interval_{Intinterval}.json"
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
    )

    PLot = False
    augment_save_dir = "data/validate_croped"
    crop_times = 8
    for _ in tqdm(range(crop_times), position=1, colour="red"):
        random_string = generate_random_string(8)
        for i in tqdm(
            range(len(val_dataset)), position=2, colour="blue", desc="dataset making!!!"
        ):
            data = val_dataset[i]
            if PLot:
                val_dataset.plot_frames(data, out_directory=augment_save_dir)
            else:
                val_dataset.make_dataset(
                    data,
                    out_directory=augment_save_dir,
                    random_string=random_string,
                )


# ----------------------------------------------------------------
# ls -l data/DLDATA/H8JPEG_valid_aug_256/*.jpg | wc -l
if __name__ == "__main__":
    # process_dataset_maker(PLot=True, flow_type="opencv_flow")
    # augment_save_dir = process_dataset_maker(PLot=False, flow_type="none")
    # augment_save_dir = process_dataset_maker(PLot=False, flow_type="opencv_flow")

    # # process_flo_dataset(flow_type="opencv_flow", data_dir=augment_save_dir)
    # splits = process_pkldataset_maker()
    # from src.utils import show_video_line
    # import pickle

    # splits = ["OhbVrpoi_trian_256_interval-19_pre_slen-6"]
    # # load the dataset
    # split = splits[0]
    # pkl_fp = f"data/{split}_cloud.pkl"
    # with open(pkl_fp, 'rb') as f:
    #     dataset = pickle.load(f)

    # train_x, train_y = dataset[f'X_{split}'], dataset[f'Y_{split}']
    # train_x, train_y = train_x/255., train_y/255.
    # print(train_x.shape)
    # # the shape is B x T x C x H x W
    # # B: the number of samples
    # # T: the number of frames in each sample
    # # C, H, W: the height, width, channel of each frame

    # # show the given frames from an example
    # example_idx = 0
    # show_video_line(train_x[example_idx], ncols=6, vmax=0.6, cbar=False, out_path="data/compare.png", format='png', use_rgb=True)

    tranvaltest_json_split()
